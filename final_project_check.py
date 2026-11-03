
from pathlib import Path
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0


def check_file(relative_path, required=True):
    """Check whether a file exists and is not empty."""
    global PASS_COUNT, FAIL_COUNT, WARN_COUNT

    path = ROOT / relative_path

    if path.is_file() and path.stat().st_size > 0:
        print(f"[PASS] {relative_path}")
        PASS_COUNT += 1
        return True

    if required:
        print(f"[FAIL] Missing or empty: {relative_path}")
        FAIL_COUNT += 1
    else:
        print(f"[WARN] Optional file missing: {relative_path}")
        WARN_COUNT += 1

    return False


def check_json(relative_path):
    """Check that a JSON file exists and contains valid JSON."""
    global PASS_COUNT, FAIL_COUNT

    path = ROOT / relative_path

    if not path.is_file():
        print(f"[FAIL] Missing JSON: {relative_path}")
        FAIL_COUNT += 1
        return

    try:
        with path.open("r", encoding="utf-8") as file:
            json.load(file)

        print(f"[PASS] Valid JSON: {relative_path}")
        PASS_COUNT += 1

    except (json.JSONDecodeError, OSError) as exc:
        print(f"[FAIL] Invalid JSON: {relative_path} ({exc})")
        FAIL_COUNT += 1


def check_jsonl(relative_path):
    """Check that each non-empty line in a JSONL file is valid JSON."""
    global PASS_COUNT, FAIL_COUNT

    path = ROOT / relative_path

    if not path.is_file():
        print(f"[FAIL] Missing JSONL: {relative_path}")
        FAIL_COUNT += 1
        return

    try:
        with path.open("r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        for line_number, line in enumerate(lines, start=1):
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on non-empty record {line_number}: {exc}"
                ) from exc

        print(
            f"[PASS] Valid JSONL: {relative_path} "
            f"({len(lines)} records)"
        )
        PASS_COUNT += 1

    except (OSError, ValueError) as exc:
        print(f"[FAIL] Invalid JSONL: {relative_path} ({exc})")
        FAIL_COUNT += 1


def check_csv(relative_path, required=True):
    """Check that a CSV exists, is non-empty, and has a header."""
    import csv

    global PASS_COUNT, FAIL_COUNT, WARN_COUNT

    path = ROOT / relative_path

    if not path.is_file() or path.stat().st_size == 0:
        if required:
            print(f"[FAIL] Missing or empty CSV: {relative_path}")
            FAIL_COUNT += 1
        else:
            print(f"[WARN] Optional CSV missing: {relative_path}")
            WARN_COUNT += 1
        return

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            header = next(reader, None)

        if header and any(column.strip() for column in header):
            print(f"[PASS] CSV header present: {relative_path}")
            PASS_COUNT += 1
        else:
            print(f"[FAIL] CSV has no usable header: {relative_path}")
            FAIL_COUNT += 1

    except (OSError, UnicodeError, csv.Error) as exc:
        print(f"[FAIL] Cannot read CSV: {relative_path} ({exc})")
        FAIL_COUNT += 1


def check_database():
    """Verify the required SQLite tables exist and contain records."""
    global PASS_COUNT, FAIL_COUNT

    db_path = ROOT / "data" / "instamart.db"

    print("\nDATABASE CHECK")

    if not db_path.is_file():
        print("[FAIL] Database not found: data/instamart.db")
        FAIL_COUNT += 1
        return

    try:
        with sqlite3.connect(db_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    """
                ).fetchall()
            }

            expected_tables = {"customers", "products", "orders"}
            missing_tables = expected_tables - tables

            if not missing_tables:
                print("[PASS] Required database tables exist")
                PASS_COUNT += 1
            else:
                print(
                    f"[FAIL] Missing database tables: "
                    f"{sorted(missing_tables)}"
                )
                FAIL_COUNT += 1

            for table in sorted(expected_tables & tables):
                count = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]

                print(f"       {table}: {count:,} rows")

                if count > 0:
                    print(f"[PASS] {table} contains data")
                    PASS_COUNT += 1
                else:
                    print(f"[FAIL] {table} is empty")
                    FAIL_COUNT += 1

    except sqlite3.Error as exc:
        print(f"[FAIL] Database error: {exc}")
        FAIL_COUNT += 1


def main():
    global PASS_COUNT, FAIL_COUNT, WARN_COUNT

    print("=" * 65)
    print("INSTAMART CAPSTONE — FINAL PROJECT AUDIT")
    print("=" * 65)
    print(f"Project root: {ROOT}")

    # ---------------------------------------------------------
    # Core source files
    # ---------------------------------------------------------
    print("\nCORE FILES")

    core_files = [
        "generate_data.py",
        "mock_llm.py",
        "clean_data.py",
        "build_database.py",
        "sql/queries.sql",
        "part_c_ticket_triage.py",
        "part_d_rag.py",
        "part_d_refund_agent.py",
    ]

    for item in core_files:
        check_file(item)

    # ---------------------------------------------------------
    # Part A — Data Foundation and SQL Analytics
    # ---------------------------------------------------------
    print("\nPART A — DATA FOUNDATION")

    check_file("data/instamart.db")

    # Correct filenames based on the user's actual EDA folder.
    eda_files = [
        "outputs/eda/eda_findings.md",
        "outputs/eda/descriptive_statistics.csv",
        "outputs/eda/correlation_matrix.csv",
        "outputs/eda/correlation_matrix.json",
        "outputs/eda/dataset_info.json",
        "outputs/eda/iqr_outlier_analysis.json",
        "outputs/eda/pandas_sql_crosscheck.json",
        "outputs/eda/pivot_reconciliation.json",
        "outputs/eda/revenue_pivot_city_category.csv",
    ]

    for item in eda_files:
        check_file(item)

    sql_files = [
        f"outputs/sql_results/q{i}_results.csv"
        for i in range(1, 8)
    ]

    for item in sql_files:
        check_csv(item)

    check_file(
        "outputs/sql_results/execution_summary.json",
        required=False,
    )
    check_json("outputs/sql_results/execution_summary.json")

    # ---------------------------------------------------------
    # Part B — Customer Segmentation and Lapse Prediction
    # ---------------------------------------------------------
    print("\nPART B — CUSTOMER INTELLIGENCE")

    part_b_files = [
        "outputs/customer_segments.csv",
        "outputs/lapse_model.joblib",
    ]

    for item in part_b_files:
        check_file(item)

    # ---------------------------------------------------------
    # Part C — Support Ticket Triage
    # ---------------------------------------------------------
    print("\nPART C — TICKET TRIAGE")

    part_c_files = [
        "outputs/triage_results.csv",
        "outputs/triage_budget_demo.csv",
        "outputs/triage_summary.json",
    ]

    for item in part_c_files:
        check_file(item)

    check_json("outputs/triage_summary.json")

    # ---------------------------------------------------------
    # Part D1 — Policy Retrieval-Augmented Generation
    # ---------------------------------------------------------
    print("\nPART D1 — POLICY RAG")

    rag_files = [
        "outputs/rag/policy_passages.csv",
        "outputs/rag/policy_rag_eval.csv",
        "outputs/rag/policy_rag_summary.json",
    ]

    for item in rag_files:
        check_file(item)

    check_json("outputs/rag/policy_rag_summary.json")

    # ---------------------------------------------------------
    # Part D2 — Safe Refund Agent
    # ---------------------------------------------------------
    print("\nPART D2 — SAFE REFUND AGENT")

    agent_files = [
        "outputs/agent_scenario_results.csv",
        "outputs/agent_runs.jsonl",
        "outputs/refund_ledger.jsonl",
        "outputs/agent_summary.json",
        "outputs/agent_validation.json",
    ]

    for item in agent_files:
        check_file(item)

    check_csv("outputs/agent_scenario_results.csv")
    check_jsonl("outputs/agent_runs.jsonl")
    check_jsonl("outputs/refund_ledger.jsonl")
    check_json("outputs/agent_summary.json")
    check_json("outputs/agent_validation.json")

    # ---------------------------------------------------------
    # Database integrity
    # ---------------------------------------------------------
    check_database()

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL AUDIT SUMMARY")
    print("=" * 65)
    print(f"Passed checks: {PASS_COUNT}")
    print(f"Failed checks: {FAIL_COUNT}")
    print(f"Warnings:      {WARN_COUNT}")

    if FAIL_COUNT:
        print("\nAUDIT RESULT: NOT READY — review failed checks above.")
        sys.exit(1)

    if WARN_COUNT:
        print("\nAUDIT RESULT: REQUIRED CHECKS PASSED WITH WARNINGS")
        print("Review the warnings before submission.")
    else:
        print("\nAUDIT RESULT: ALL CONFIGURED CHECKS PASSED")

    print(
        "\nNote: This audit checks file presence and basic integrity. "
        "It does not replace the individual Part A-D validators."
    )


if __name__ == "__main__":
    main()
