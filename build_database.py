
from pathlib import Path
import sqlite3
import pandas as pd

CLEAN_DIR = Path("data/clean")
DB_PATH = Path("data/instamart.db")

def main():
    tables = ["customers", "products", "orders"]

    # Verify all required CSVs exist.
    for table in tables:
        path = CLEAN_DIR / f"{table}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    # Remove the old database so obsolete tables cannot remain.
    if DB_PATH.exists():
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        for table in tables:
            df = pd.read_csv(CLEAN_DIR / f"{table}.csv")

            if table == "orders":
                df["order_ts"] = df["order_ts"].astype(str)

            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"Loaded {table}: {len(df)} rows")

        conn.execute(
            "CREATE INDEX idx_orders_customer_id ON orders(customer_id)"
        )
        conn.execute(
            "CREATE INDEX idx_orders_product_id ON orders(product_id)"
        )
        conn.execute(
            "CREATE INDEX idx_orders_status ON orders(status)"
        )

        actual_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

        print("\nDatabase tables:", sorted(actual_tables))

        assert actual_tables == set(tables), (
            f"Unexpected tables: {sorted(actual_tables)}"
        )

        expected = {"customers": 1200, "products": 40, "orders": 12309}

        for table, expected_count in expected.items():
            count = conn.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            print(f"{table}: {count} rows")
            assert count == expected_count

    print("\nALL DATABASE CHECKS PASSED")
    print(f"Database saved at: {DB_PATH.resolve()}")

if __name__ == "__main__":
    main()
