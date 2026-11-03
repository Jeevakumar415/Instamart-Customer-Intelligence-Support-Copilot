
from pathlib import Path
import json
import pandas as pd

ROOT = Path(".")
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs" / "eda"
REPORT_PATH = OUTPUT_DIR / "eda_findings.md"


def read_json(path):
    """Read a JSON file if it exists."""
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def money(value):
    """Format a numeric value as Indian rupees."""
    return f"₹{float(value):,.2f}"


def get_raw_audit_rows(raw_audit):
    """Support both list-based and dictionary-based audit JSON."""
    if isinstance(raw_audit, list):
        result = []

        for index, item in enumerate(raw_audit):
            if not isinstance(item, dict):
                continue

            dataset = (
                item.get("dataset")
                or item.get("file")
                or item.get("filename")
                or item.get("name")
                or f"Dataset {index + 1}"
            )
            result.append((dataset, item))

        return result

    if isinstance(raw_audit, dict):
        # Some audit formats store datasets under a "datasets" key.
        nested = raw_audit.get("datasets")

        if isinstance(nested, list):
            return get_raw_audit_rows(nested)

        if isinstance(nested, dict):
            return list(nested.items())

        # A single audit record may itself be a dictionary.
        if any(
            key in raw_audit
            for key in ("rows", "row_count", "file", "filename")
        ):
            dataset = (
                raw_audit.get("dataset")
                or raw_audit.get("file")
                or raw_audit.get("filename")
                or "Dataset"
            )
            return [(dataset, raw_audit)]

        # Typical format: dataset name -> audit details.
        return [
            (name, details)
            for name, details in raw_audit.items()
            if isinstance(details, dict)
        ]

    return []


def format_missing_values(missing):
    """Turn common missing-value audit structures into readable text."""
    items = []

    if isinstance(missing, dict):
        for column, value in missing.items():
            if isinstance(value, dict):
                count = value.get("count", value.get("missing_count", 0))
                if count:
                    items.append(f"{column}: {count}")
            elif isinstance(value, (int, float)):
                if value != 0:
                    items.append(f"{column}: {value}")
            elif value not in (None, "", "0", False):
                items.append(f"{column}: {value}")

    elif isinstance(missing, list):
        for item in missing:
            if isinstance(item, dict):
                column = item.get("column", item.get("name", "column"))
                count = item.get("count", item.get("missing_count", "N/A"))
                items.append(f"{column}: {count}")
            else:
                items.append(str(item))

    return ", ".join(items) if items else "See raw audit JSON"


def build_raw_audit_table(raw_audit):
    """Create Markdown rows from the raw-data audit."""
    rows = []

    for dataset, details in get_raw_audit_rows(raw_audit):
        count = details.get(
            "rows",
            details.get("row_count", details.get("total_rows", "N/A")),
        )

        duplicates = details.get(
            "exact_duplicate_rows",
            details.get("duplicate_rows", details.get("duplicates", "N/A")),
        )

        missing = details.get(
            "missing_values",
            details.get("missing", details.get("missing_data", {})),
        )

        missing_text = format_missing_values(missing)

        rows.append(
            f"| {dataset} | {count} | {duplicates} | {missing_text} |"
        )

    return "\n".join(rows) or (
        "| Details unavailable | See audit JSON | See audit JSON "
        "| Inspect outputs/raw_data_audit.json |"
    )


def build_cleaning_table(cleaning):
    """Create Markdown rows from the cleaning report."""
    rows = []

    if not isinstance(cleaning, dict):
        return "| Details unavailable | N/A | N/A |"

    for dataset, details in cleaning.items():
        if not isinstance(details, dict):
            continue

        input_rows = details.get(
            "input_rows",
            details.get("rows_before", details.get("input", "N/A")),
        )
        output_rows = details.get(
            "output_rows",
            details.get("rows_after", details.get("output", "N/A")),
        )

        rows.append(f"| {dataset} | {input_rows} | {output_rows} |")

    return "\n".join(rows) or (
        "| Details unavailable | See cleaning report | See cleaning report |"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        CLEAN_DIR / "customers.csv",
        CLEAN_DIR / "products.csv",
        CLEAN_DIR / "orders.csv",
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Required cleaned datasets are missing:\n"
            + "\n".join(missing_files)
            + "\nRun your data cleaning pipeline first."
        )

    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")

    required_order_columns = {
        "customer_id",
        "product_id",
        "status",
        "order_amount",
        "quantity",
        "discount_pct",
        "delivery_minutes",
    }

    absent = required_order_columns - set(orders.columns)

    if absent:
        raise ValueError(
            f"Orders dataset is missing required columns: {sorted(absent)}"
        )

    for column in [
        "order_amount",
        "quantity",
        "discount_pct",
        "delivery_minutes",
    ]:
        orders[column] = pd.to_numeric(orders[column], errors="coerce")

    orders["status"] = orders["status"].astype("string").str.strip().str.lower()

    completed = orders.loc[orders["status"].eq("completed")].copy()

    if completed.empty:
        raise ValueError("No completed orders found; cannot produce revenue EDA.")

    # Load generated audit and EDA outputs.
    raw_audit = read_json(ROOT / "outputs" / "raw_data_audit.json")
    cleaning = read_json(ROOT / "outputs" / "cleaning_report.json")
    crosscheck = read_json(OUTPUT_DIR / "pandas_sql_crosscheck.json")
    pivot_check = read_json(OUTPUT_DIR / "pivot_reconciliation.json")

    # Merge completed orders with city and category.
    order_details = (
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

    if order_details["city"].isna().any():
        print("Warning: Some completed orders have no matching customer city.")

    if order_details["category"].isna().any():
        print("Warning: Some completed orders have no matching product category.")

    # Revenue aggregations.
    city_revenue = (
        order_details.groupby("city", dropna=False)["order_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    category_revenue = (
        order_details.groupby("category", dropna=False)["order_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    city_category = (
        order_details.groupby(["city", "category"], dropna=False)["order_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    city_lines = [
        f"| {city} | {money(revenue)} |"
        for city, revenue in city_revenue.items()
    ]

    category_lines = [
        f"| {category} | {money(revenue)} |"
        for category, revenue in category_revenue.items()
    ]

    city_category_lines = [
        f"| {city} | {category} | {money(revenue)} |"
        for (city, category), revenue in city_category.head(10).items()
    ]

    # Descriptive statistics.
    numeric_columns = [
        "quantity",
        "discount_pct",
        "order_amount",
        "delivery_minutes",
    ]

    descriptive = completed[numeric_columns].describe(
        percentiles=[0.25, 0.5, 0.75]
    )

    descriptive_lines = []

    for statistic, row in descriptive.iterrows():
        values = [
            f"{value:.2f}" if pd.notna(value) else "N/A"
            for value in row
        ]
        descriptive_lines.append(
            f"| {statistic} | " + " | ".join(values) + " |"
        )

    # IQR analysis. Flag, but do not automatically remove, outliers.
    outlier_lines = []

    for column in ["order_amount", "delivery_minutes"]:
        values = completed[column].dropna()

        if values.empty:
            outlier_lines.append(
                f"| {column} | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = (values < lower) | (values > upper)
        outlier_count = int(outliers.sum())
        outlier_percentage = float(outliers.mean() * 100)

        outlier_lines.append(
            f"| {column} | {q1:.2f} | {q3:.2f} | {iqr:.2f} "
            f"| {lower:.2f} | {upper:.2f} "
            f"| {outlier_count:,} ({outlier_percentage:.2f}%) |"
        )

    # Correlation matrix.
    correlation_path = OUTPUT_DIR / "correlation_matrix.csv"

    if correlation_path.exists():
        correlation = pd.read_csv(correlation_path, index_col=0)
    else:
        correlation = completed[numeric_columns].corr(method="pearson")

    correlation_columns = list(correlation.columns)
    correlation_header = " | ".join(correlation_columns)

    correlation_lines = []

    for variable, row in correlation.iterrows():
        values = [
            "N/A" if pd.isna(value) else f"{value:.3f}"
            for value in row
        ]
        correlation_lines.append(
            f"| {variable} | " + " | ".join(values) + " |"
        )

    # Order status counts.
    status_counts = orders["status"].value_counts(dropna=False)

    status_lines = [
        f"| {status if pd.notna(status) else 'Unknown'} | {count:,} |"
        for status, count in status_counts.items()
    ]

    # Revenue totals.
    completed_revenue = float(completed["order_amount"].sum())
    completed_count = len(completed)
    average_order_amount = float(completed["order_amount"].mean())

    pandas_revenue = crosscheck.get(
        "pandas_completed_revenue", completed_revenue
    )
    sqlite_revenue = crosscheck.get("sqlite_completed_revenue")
    crosscheck_difference = crosscheck.get("difference", "Not recorded")
    crosscheck_match = crosscheck.get("match", False)

    pivot_total = pivot_check.get("pivot_grand_total")
    pivot_expected = pivot_check.get(
        "expected_completed_revenue", completed_revenue
    )
    pivot_match = pivot_check.get("match", False)

    # Generate report.
    report = f"""# Part A — Data Foundation, SQL Analytics and EDA

## 1. Objective

This part establishes a reproducible data foundation for the
Instamart Customer Intelligence & Support Copilot capstone.
It audits and cleans the supplied synthetic datasets, builds a
SQLite database, executes the required SQL analyses, and checks
the consistency of revenue calculations.

This is a synthetic educational project and is not an official
Instamart system.

## 2. Dataset overview

| Dataset | Clean rows | Number of columns |
|---|---:|---:|
| Customers | {len(customers):,} | {len(customers.columns)} |
| Products | {len(products):,} | {len(products.columns)} |
| Orders | {len(orders):,} | {len(orders.columns)} |

### Raw-data audit summary

| Dataset | Raw rows | Exact duplicate rows | Missing-value notes |
|---|---:|---:|---|
{build_raw_audit_table(raw_audit)}

Full raw-data audit: `outputs/raw_data_audit.json`

## 3. Data cleaning

The cleaning pipeline writes reusable datasets under `data/clean/`
and records its results in `outputs/cleaning_report.json`.

Implemented steps include:

- Normalizing customer city values and gender labels.
- Treating ages outside the configured 10–100 range as invalid
  and imputing missing ages.
- Removing exact duplicate order rows.
- Keeping the latest record for each order ID.
- Converting dates and numeric fields into usable types.
- Filling missing or invalid order amounts using product price,
  quantity and discount where applicable.
- Setting negative delivery durations to missing.
- Imputing missing delivery durations for completed orders using
  the median completed delivery duration for the product category.
- Keeping cancelled orders' delivery durations missing.
- Copying ticket, policy evaluation and agent scenario data into
  the clean-data directory.

Imputed values are estimates, not original observations.

### Cleaning row counts

| Dataset | Input rows | Output rows |
|---|---:|---:|
{build_cleaning_table(cleaning)}

## 4. Order status and revenue

| Order status | Rows |
|---|---:|
{chr(10).join(status_lines)}

- Completed orders: **{completed_count:,}**
- Completed-order revenue: **{money(completed_revenue)}**
- Average completed order amount: **{money(average_order_amount)}**

Revenue calculations in this section use completed orders only.

## 5. Revenue by city

| City | Completed-order revenue |
|---|---:|
{chr(10).join(city_lines)}

## 6. Revenue by product category

| Product category | Completed-order revenue |
|---|---:|
{chr(10).join(category_lines)}

### Top 10 city-category combinations

| City | Category | Revenue |
|---|---|---:|
{chr(10).join(city_category_lines)}

## 7. Descriptive statistics

The following statistics use completed orders and the numeric
columns quantity, discount percentage, order amount and delivery
minutes.

| Statistic | Quantity | Discount % | Order amount | Delivery minutes |
|---|---:|---:|---:|---:|
{chr(10).join(descriptive_lines)}

## 8. IQR outlier analysis

Outliers are flagged using the 1.5 × IQR rule. They are not
automatically removed.

| Variable | Q1 | Q3 | IQR | Lower bound | Upper bound | Flagged outliers |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(outlier_lines)}

An outlier can be a valid high-value order or an unusually long
delivery. Domain review is needed before excluding observations.

## 9. Correlation analysis

Pearson correlation matrix for the selected numeric variables:

| Variable | {correlation_header} |
|---|---|
{chr(10).join(correlation_lines)}

Correlation describes linear association; it does not establish
causation. Interpret these values in the context of the synthetic
data and the way missing values were handled.

## 10. Revenue reconciliation

### Pandas versus SQLite

- Pandas completed-order revenue: **{money(pandas_revenue)}**
- SQLite completed-order revenue: **{money(sqlite_revenue) if sqlite_revenue is not None else 'Not available'}**
- Difference: **{crosscheck_difference}**
- Match recorded by analysis: **{crosscheck_match}**

### Pivot-table reconciliation

- Pivot grand total: **{money(pivot_total) if pivot_total is not None else 'Not available'}**
- Expected completed revenue: **{money(pivot_expected)}**
- Match recorded by analysis: **{pivot_match}**

The current analysis reports matching revenue totals of
{money(completed_revenue)}. The separate validation script should
also be run to confirm all required checks.

## 11. SQL analytics deliverables

The seven query results are stored in `outputs/sql_results/`:

1. Filtering and grouped customer spending.
2. Multi-table revenue aggregation by city and category.
3. Products priced above the average.
4. Customers without completed orders.
5. Customers spending above the average customer total.
6. Customer ranking within cities using window functions.
7. Daily revenue with a cumulative running total.

SQL source: `sql/queries.sql`

## 12. Reproducibility and output locations

- Clean datasets: `data/clean/`
- SQLite database: `data/instamart.db`
- SQL source: `sql/queries.sql`
- SQL results: `outputs/sql_results/`
- Raw-data audit: `outputs/raw_data_audit.json`
- Cleaning report: `outputs/cleaning_report.json`
- Part A validation: `outputs/part_a_validation.json`
- EDA artifacts and this report: `outputs/eda/`

## 13. Conclusion

Part A has produced cleaned datasets, a SQLite database, seven
SQL query outputs, audit and cleaning reports, descriptive
statistics, an IQR analysis, a correlation matrix, and revenue
reconciliation checks.

The final project-level validation must be rerun before declaring
Part A complete, to confirm that all expected artifacts remain
present and consistent.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("=" * 65)
    print("PART A — EDA FINDINGS REPORT GENERATED")
    print("=" * 65)
    print(f"Report: {REPORT_PATH.resolve()}")
    print(f"Completed orders: {completed_count:,}")
    print(f"Completed revenue: {money(completed_revenue)}")
    print(f"Pandas/SQLite match: {crosscheck_match}")
    print(f"Pivot reconciliation: {pivot_match}")


if __name__ == "__main__":
    main()
