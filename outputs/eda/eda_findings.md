# Part A — Data Foundation, SQL Analytics and EDA

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
| Customers | 1,200 | 5 |
| Products | 40 | 4 |
| Orders | 12,309 | 9 |

### Raw-data audit summary

| Dataset | Raw rows | Exact duplicate rows | Missing-value notes |
|---|---:|---:|---|
| customers.csv | 1200 | 0 | customer_id: 0, city: 0, signup_date: 0, age: 61, gender: 30 |
| products.csv | 40 | 0 | product_id: 0, product_name: 0, category: 0, price_inr: 0 |
| orders.csv | 13084 | 119 | order_id: 0, customer_id: 0, product_id: 0, quantity: 0, order_ts: 0, status: 0, discount_pct: 0, order_amount: 53, delivery_minutes: 2541 |
| tickets.csv | 164 | 0 | ticket_id: 0, customer_id: 0, order_id: 8, text: 4, true_category: 0 |
| policy_eval.csv | 14 | 0 | question: 0, expected_doc: 0, expected_fact: 2 |
| agent_scenarios.csv | 7 | 0 | scenario_id: 0, customer_id: 0, message: 0, expected_outcome: 0, expected_refund_inr: 0 |

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
| customers | 1200 | 1200 |
| products | N/A | 40 |
| orders | 13084 | 12309 |
| other_inputs | N/A | N/A |

## 4. Order status and revenue

| Order status | Rows |
|---|---:|
| completed | 11,110 |
| cancelled | 1,199 |

- Completed orders: **11,110**
- Completed-order revenue: **₹2,998,668.80**
- Average completed order amount: **₹269.91**

Revenue calculations in this section use completed orders only.

## 5. Revenue by city

| City | Completed-order revenue |
|---|---:|
| Hyderabad | ₹528,589.90 |
| Bengaluru | ₹523,643.40 |
| Delhi | ₹502,829.80 |
| Pune | ₹500,918.05 |
| Mumbai | ₹479,058.50 |
| Chennai | ₹463,629.15 |

## 6. Revenue by product category

| Product category | Completed-order revenue |
|---|---:|
| Electronics | ₹1,219,248.10 |
| Grocery | ₹614,293.30 |
| Dairy | ₹531,938.95 |
| Snacks | ₹464,754.50 |
| Personal Care | ₹168,433.95 |

### Top 10 city-category combinations

| City | Category | Revenue |
|---|---|---:|
| Bengaluru | Electronics | ₹233,909.45 |
| Hyderabad | Electronics | ₹228,913.10 |
| Mumbai | Electronics | ₹214,738.40 |
| Chennai | Electronics | ₹195,508.80 |
| Pune | Electronics | ₹187,629.25 |
| Delhi | Electronics | ₹158,549.10 |
| Delhi | Dairy | ₹117,271.70 |
| Delhi | Grocery | ₹112,658.05 |
| Bengaluru | Grocery | ₹110,915.65 |
| Hyderabad | Grocery | ₹109,475.40 |

## 7. Descriptive statistics

The following statistics use completed orders and the numeric
columns quantity, discount percentage, order amount and delivery
minutes.

| Statistic | Quantity | Discount % | Order amount | Delivery minutes |
|---|---:|---:|---:|---:|
| count | 11110.00 | 11110.00 | 11110.00 | 11110.00 |
| mean | 1.89 | 12.96 | 269.91 | 27.37 |
| std | 0.78 | 13.05 | 432.07 | 12.75 |
| min | 1.00 | 0.00 | 12.00 | 8.00 |
| 25% | 1.00 | 0.00 | 64.80 | 20.00 |
| 50% | 2.00 | 10.00 | 135.00 | 25.00 |
| 75% | 3.00 | 20.00 | 297.00 | 31.00 |
| max | 3.00 | 40.00 | 3998.00 | 75.00 |

## 8. IQR outlier analysis

Outliers are flagged using the 1.5 × IQR rule. They are not
automatically removed.

| Variable | Q1 | Q3 | IQR | Lower bound | Upper bound | Flagged outliers |
|---|---:|---:|---:|---:|---:|---:|
| order_amount | 64.80 | 297.00 | 232.20 | -283.50 | 645.30 | 897 (8.07%) |
| delivery_minutes | 20.00 | 31.00 | 11.00 | 3.50 | 47.50 | 919 (8.27%) |

An outlier can be a valid high-value order or an unusually long
delivery. Domain review is needed before excluding observations.

## 9. Correlation analysis

Pearson correlation matrix for the selected numeric variables:

| Variable | quantity | discount_pct | order_amount | delivery_minutes |
|---|---|
| quantity | 1.000 | -0.000 | 0.087 | -0.013 |
| discount_pct | -0.000 | 1.000 | -0.095 | -0.010 |
| order_amount | 0.087 | -0.095 | 1.000 | -0.003 |
| delivery_minutes | -0.013 | -0.010 | -0.003 | 1.000 |

Correlation describes linear association; it does not establish
causation. Interpret these values in the context of the synthetic
data and the way missing values were handled.

## 10. Revenue reconciliation

### Pandas versus SQLite

- Pandas completed-order revenue: **₹2,998,668.80**
- SQLite completed-order revenue: **₹2,998,668.80**
- Difference: **0.0**
- Match recorded by analysis: **True**

### Pivot-table reconciliation

- Pivot grand total: **₹2,998,668.80**
- Expected completed revenue: **₹2,998,668.80**
- Match recorded by analysis: **True**

The current analysis reports matching revenue totals of
₹2,998,668.80. The separate validation script should
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
