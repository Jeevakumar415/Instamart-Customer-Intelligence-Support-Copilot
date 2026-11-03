
from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path("data/instamart.db")
SQL_PATH = Path("sql/queries.sql")
OUTPUT_DIR = Path("outputs/sql_results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_queries():
    """Load the seven SQL queries from the assignment SQL file."""
    if not SQL_PATH.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_PATH}")

    sql_text = SQL_PATH.read_text(encoding="utf-8")

    queries = [
        query.strip()
        for query in sql_text.split(";")
        if query.strip()
    ]

    if len(queries) != 7:
        raise ValueError(
            f"Expected 7 SQL queries, but found {len(queries)}. "
            "Check sql/queries.sql."
        )

    return queries


def print_result(query_number, result):
    """Print and save each query's complete result."""
    print("\n" + "=" * 65)
    print(f"Q{query_number} RESULTS")
    print("=" * 65)

    print(f"Rows returned: {len(result)}")
    print(result.head(10).to_string(index=False))

    output_file = OUTPUT_DIR / f"q{query_number}_results.csv"
    result.to_csv(output_file, index=False)

    print(f"Full results saved to: {output_file}")


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. "
            "Run python build_database.py first."
        )

    queries = load_queries()

    with sqlite3.connect(DB_PATH) as connection:
        # Verify the database contains the required three tables.
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        expected_tables = {"customers", "products", "orders"}

        if tables != expected_tables:
            raise RuntimeError(
                f"Expected tables {sorted(expected_tables)}, "
                f"but found {sorted(tables)}."
            )

        print("Database verified:", DB_PATH)
        print("Tables:", ", ".join(sorted(tables)))

        results = {}

        # Execute all seven assignment queries.
        for index, query in enumerate(queries, start=1):
            result = pd.read_sql_query(query, connection)
            results[index] = result
            print_result(index, result)

        # Q1: WHERE followed by HAVING.
        print("\nQ1 CHECK")
        print("Customer count:", len(results[1]))
        print("Reference customer count: 30")

        # Q2: Revenue by city and category.
        q2 = results[2]
        if "revenue_inr" in q2.columns:
            print("\nQ2 CHECK")
            print(
                "Grand total revenue (INR):",
                round(q2["revenue_inr"].sum(), 2),
            )

            largest_cell = q2.loc[q2["revenue_inr"].idxmax()]
            print(
                "Largest revenue cell:",
                largest_cell["city"],
                "/",
                largest_cell["category"],
                "=",
                round(largest_cell["revenue_inr"], 2),
            )

        # Q3: Products above the average catalogue price.
        print("\nQ3 CHECK")
        print("Products returned:", len(results[3]))
        print("Reference product count: 11")

        # Q4: Customers without a completed order.
        print("\nQ4 CHECK")
        print("Customers returned:", len(results[4]))
        print("Reference customer count: 25")

        # Q5: Average must be computed from ALL customers with completed
        # orders, not from the 550 customers above the average threshold.
        q5 = results[5]
        all_customer_totals = pd.read_sql_query(
            """
            SELECT
                customer_id,
                SUM(order_amount) AS total_spend
            FROM orders
            WHERE status = 'completed'
            GROUP BY customer_id
            """,
            connection,
        )

        overall_average = all_customer_totals["total_spend"].mean()

        print("\nQ5 CHECK")
        print("Customers above average:", len(q5))
        print(
            "Average completed spend across all customers with completed "
            f"orders (INR): {overall_average:.2f}"
        )
        print("Reference customer count: 550")
        print("Reference average (INR): 2552.06")

        # Q6: Check rankings and city leaders.
        q6 = results[6]
        print("\nQ6 CHECK")

        if not q6.empty:
            rank_column = "spend_rank"
            row_number_column = "spend_row_number"

            if rank_column in q6.columns and row_number_column in q6.columns:
                different_ranks = int(
                    (q6[rank_column] != q6[row_number_column]).sum()
                )
                print("Customers ranked:", len(q6))
                print(
                    "Rows where RANK differs from ROW_NUMBER:",
                    different_ranks,
                )
                print("Reference difference count: 8")

                city_leaders = (
                    q6[q6[rank_column] == 1]
                    .sort_values("city")
                )

                print("\nTop-ranked customers by city:")
                print(
                    city_leaders[
                        ["city", "customer_id", "total_spend"]
                    ].to_string(index=False)
                )

        # Q7: Daily revenue running total.
        q7 = results[7]
        print("\nQ7 CHECK")
        print("Daily rows:", len(q7))

        if not q7.empty and "cumulative_revenue_inr" in q7.columns:
            print(
                "Final cumulative revenue (INR):",
                f"{q7['cumulative_revenue_inr'].iloc[-1]:.2f}",
            )

        # Save an execution summary.
        summary = {
            "q1_rows": len(results[1]),
            "q2_rows": len(results[2]),
            "q3_rows": len(results[3]),
            "q4_rows": len(results[4]),
            "q5_rows": len(results[5]),
            "q6_rows": len(results[6]),
            "q7_rows": len(results[7]),
            "q2_revenue_inr": round(
                results[2]["revenue_inr"].sum(), 2
            ),
            "q5_overall_average_inr": round(overall_average, 2),
            "q7_final_cumulative_revenue_inr": (
                round(q7["cumulative_revenue_inr"].iloc[-1], 2)
                if not q7.empty
                else 0
            ),
        }

        summary_file = OUTPUT_DIR / "execution_summary.json"
        import json

        summary_file.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 65)
        print("ALL SEVEN SQL QUERIES EXECUTED SUCCESSFULLY")
        print("=" * 65)
        print(f"Query CSVs: {OUTPUT_DIR.resolve()}")
        print(f"Summary: {summary_file.resolve()}")


if __name__ == "__main__":
    main()
