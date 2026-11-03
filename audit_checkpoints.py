
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")


def main():
    orders = pd.read_csv(
        RAW_DIR / "orders.csv",
        keep_default_na=False,
    )
    customers = pd.read_csv(
        RAW_DIR / "customers.csv",
        keep_default_na=False,
    )

    # Order audit
    numeric_amount = pd.to_numeric(
        orders["order_amount"].replace("", pd.NA),
        errors="coerce",
    )

    delivery = pd.to_numeric(
        orders["delivery_minutes"].replace("", pd.NA),
        errors="coerce",
    )

    print("RAW ORDER CHECKPOINTS")
    print("Rows:", len(orders))
    print("Exact duplicate rows:", int(orders.duplicated().sum()))
    print(
        "Pending rows:",
        int(
            orders["status"].astype(str).str.strip().str.lower()
            .eq("pending").sum()
        ),
    )
    print("Distinct order_id:", orders["order_id"].nunique())
    print(
        "Non-numeric or blank order_amount:",
        int(numeric_amount.isna().sum()),
    )
    print("Negative delivery times:", int((delivery < 0).sum()))
    print("Blank delivery times:", int(delivery.isna().sum()))

    # Customer audit
    ages = pd.to_numeric(
        customers["age"].replace("", pd.NA),
        errors="coerce",
    )
    cities = customers["city"].astype(str).str.strip()
    genders = customers["gender"].astype(str).str.strip()

    print("\nRAW CUSTOMER CHECKPOINTS")
    print("Blank ages:", int(ages.isna().sum()))
    print(
        "Ages outside 10-100:",
        int((ages.notna() & ~ages.between(10, 100)).sum()),
    )
    print("Distinct raw city spellings:", cities.nunique())
    print(
        "Distinct non-blank raw gender spellings:",
        genders[genders.ne("")].nunique(),
    )
    print("Blank genders:", int(genders.eq("").sum()))


if __name__ == "__main__":
    main()
