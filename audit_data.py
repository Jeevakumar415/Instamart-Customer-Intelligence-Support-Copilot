
from pathlib import Path
import json
import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "customers.csv",
    "products.csv",
    "orders.csv",
    "tickets.csv",
    "policy_eval.csv",
    "agent_scenarios.csv",
]


def audit_file(path):
    df = pd.read_csv(path, keep_default_na=True)
    total_rows = len(df)

    missing = []
    for column in df.columns:
        count = int(df[column].isna().sum())
        missing.append({
            "column": column,
            "missing_count": count,
            "missing_percent": round(
                count / total_rows * 100, 2
            ) if total_rows else 0.0,
        })

    return {
        "file": path.name,
        "rows": total_rows,
        "columns": list(df.columns),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "missing_values": missing,
    }


def main():
    reports = []

    for filename in FILES:
        path = RAW_DIR / filename

        if not path.exists():
            print(f"WARNING: Missing file: {path}")
            continue

        report = audit_file(path)
        reports.append(report)

        print(f"\n{'=' * 60}")
        print(f"FILE: {report['file']}")
        print(f"Rows: {report['rows']}")
        print(f"Exact duplicate rows: {report['exact_duplicate_rows']}")
        print("Missing-value audit:")

        for item in report["missing_values"]:
            print(
                f"  {item['column']}: "
                f"{item['missing_count']} "
                f"({item['missing_percent']}%)"
            )

    output_path = OUTPUT_DIR / "raw_data_audit.json"
    output_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nAudit saved to: {output_path}")


if __name__ == "__main__":
    main()
