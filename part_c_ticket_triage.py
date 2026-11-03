
from pathlib import Path
import json
import re
import time

import pandas as pd

import mock_llm


# ============================================================
# INSTAMART CAPSTONE — PART C
# SUPPORT TICKET TRIAGE
# ============================================================

TICKETS_PATH = Path("data/clean/tickets.csv")
OUTPUT_DIR = Path("outputs")
TRIAGE_PATH = OUTPUT_DIR / "triage_results.csv"
BUDGET_PATH = OUTPUT_DIR / "triage_budget_demo.csv"
SUMMARY_PATH = OUTPUT_DIR / "triage_summary.json"

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

# Input budget applies to the redacted ticket text.
MAX_INPUT_TOKENS = 250
MAX_RETRIES = 3
MAX_TICKET_CHARS = 12000

SYSTEM_PROMPT = """
CLASSIFY

You are a support ticket classifier.
Classify each ticket into exactly one category:
billing, delivery, technical, or fraud.

Return only a valid JSON object with these keys:
{"category": "one allowed category", "reason": "brief reason"}

Do not include markdown fences or extra prose.
Use only the allowed categories.
""".strip()


# ------------------------------------------------------------
# PII REDACTION
# ------------------------------------------------------------

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)

PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s().-]*)?"
    r"(?:\d[\s().-]*){9,12}(?!\w)"
)

# Common Indian-style identifiers and payment card patterns.
AADHAAR_RE = re.compile(
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
)

CARD_RE = re.compile(
    r"\b(?:\d[ -]*?){13,19}\b"
)

UPI_RE = re.compile(
    r"\b[a-zA-Z0-9._-]{2,}@[a-zA-Z]{2,}\b"
)

NAME_AFTER_LABEL_RE = re.compile(
    r"(?i)\b(?:customer name|full name|name)\s*[:=-]\s*"
    r"[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3}"
)


def redact_pii(text):
    """Redact common PII patterns before any model call."""
    if pd.isna(text):
        return ""

    result = str(text)

    result = EMAIL_RE.sub("[REDACTED_EMAIL]", result)
    result = UPI_RE.sub("[REDACTED_UPI]", result)
    result = AADHAAR_RE.sub("[REDACTED_ID]", result)
    result = CARD_RE.sub("[REDACTED_CARD]", result)
    result = PHONE_RE.sub("[REDACTED_PHONE]", result)
    result = NAME_AFTER_LABEL_RE.sub(
        lambda match: match.group(0).split(":")[0] + ": [REDACTED_NAME]",
        result,
    )

    return result


def truncate_to_token_budget(text, token_limit=MAX_INPUT_TOKENS):
    """
    Truncate text until mock_llm.est_tokens(text) fits the budget.
    The mock token estimator is used so the limit is reproducible.
    """
    text = str(text or "").strip()

    if mock_llm.est_tokens(text) <= token_limit:
        return text, False

    words = text.split()
    kept = []

    for word in words:
        candidate = " ".join(kept + [word])
        if mock_llm.est_tokens(candidate) > token_limit:
            break
        kept.append(word)

    return " ".join(kept), True


# ------------------------------------------------------------
# MODEL RESPONSE VALIDATION
# ------------------------------------------------------------

def extract_json_object(content):
    """Parse a JSON object; reject prose-only and non-object outputs."""
    if not content or not isinstance(content, str):
        raise ValueError("Empty model response.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Response is not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object.")

    return parsed


def validate_classification(content):
    """Validate schema, category, and reason."""
    data = extract_json_object(content)

    category = data.get("category")
    reason = data.get("reason")

    if not isinstance(category, str):
        raise ValueError("Missing or invalid category.")

    category = category.strip().lower()

    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Missing or invalid reason.")

    return category, reason.strip()


def fallback_category(text):
    """
    Deterministic local fallback used after repeated invalid responses.
    Priority follows fraud, technical, billing, delivery.
    """
    lowered = (text or "").lower()

    rules = [
        ("fraud", ["unauthorized", "fraud", "hacked", "stolen"]),
        ("technical", ["app", "crash", "login", "otp", "error", "freezes"]),
        ("billing", ["charged", "invoice", "bill", "refund", "deducted", "payment"]),
        ("delivery", ["arrived", "rider", "delivery", "delivered", "late"]),
    ]

    for category, keywords in rules:
        if any(keyword in lowered for keyword in keywords):
            return category

    return "technical"


def classify_with_retry(redacted_text):
    """
    Retry transient exceptions.
    Invalid JSON/schema responses are retried too, with a bounded attempt
    count. The fallback keeps the batch running if all attempts fail.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Classify this support ticket:\n"
                    + redacted_text
                ),
            },
        ]

        try:
            response = mock_llm.chat(messages)
            content = response.choices[0].message.content
            category, reason = validate_classification(content)

            return {
                "category": category,
                "reason": reason,
                "status": "success",
                "attempts": attempt,
                "error": "",
            }

        except mock_llm.TransientLLMError as exc:
            last_error = f"transient_error: {exc}"
            if attempt < MAX_RETRIES:
                time.sleep(0.05 * attempt)

        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            last_error = f"invalid_response: {exc}"
            if attempt < MAX_RETRIES:
                time.sleep(0.02 * attempt)

    category = fallback_category(redacted_text)

    return {
        "category": category,
        "reason": "Local fallback after model retries failed.",
        "status": "fallback",
        "attempts": MAX_RETRIES,
        "error": last_error or "Unknown model error",
    }


# ------------------------------------------------------------
# BATCH PROCESSING AND BUDGET TRACKING
# ------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TICKETS_PATH.exists():
        raise FileNotFoundError(
            f"Ticket dataset not found: {TICKETS_PATH}"
        )

    tickets = pd.read_csv(TICKETS_PATH)

    if tickets.empty:
        raise ValueError("Ticket dataset is empty.")

    # Resolve common ticket column names without changing source data.
    column_lookup = {
        str(column).strip().lower(): column
        for column in tickets.columns
    }

    ticket_id_column = next(
        (
            column_lookup[name]
            for name in ["ticket_id", "id", "case_id"]
            if name in column_lookup
        ),
        None,
    )

    text_column = next(
        (
            column_lookup[name]
            for name in [
                "text",
                "ticket_text",
                "message",
                "description",
                "issue",
            ]
            if name in column_lookup
        ),
        None,
    )

    if text_column is None:
        raise ValueError(
            "Could not identify the ticket text column. "
            f"Available columns: {list(tickets.columns)}"
        )

    if ticket_id_column is None:
        tickets["ticket_id"] = [
            f"TICKET-{index + 1:05d}"
            for index in range(len(tickets))
        ]
        ticket_id_column = "ticket_id"

    # Reset mock counters once per independent batch run.
    mock_llm.reset()

    results = []
    budget_rows = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_calls = 0
    redacted_tickets = 0
    truncated_tickets = 0
    empty_tickets = 0
    fallback_tickets = 0
    failed_tickets = 0

    for row_index, row in tickets.iterrows():
        ticket_id = str(row[ticket_id_column])
        original_text = "" if pd.isna(row[text_column]) else str(row[text_column])

        # Keep excessively large source strings bounded before regex work.
        if len(original_text) > MAX_TICKET_CHARS:
            original_text = original_text[:MAX_TICKET_CHARS]

        redacted_text = redact_pii(original_text)

        pii_changed = redacted_text != original_text
        if pii_changed:
            redacted_tickets += 1

        if not redacted_text.strip():
            empty_tickets += 1
            results.append({
                "ticket_id": ticket_id,
                "category": "technical",
                "reason": "Empty ticket; model call skipped.",
                "status": "skipped_empty",
                "attempts": 0,
                "pii_redacted": pii_changed,
                "input_truncated": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": "",
            })
            budget_rows.append({
                "ticket_id": ticket_id,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "attempts": 0,
                "status": "skipped_empty",
            })
            continue

        bounded_text, truncated = truncate_to_token_budget(redacted_text)

        if truncated:
            truncated_tickets += 1

        # Use the same exact message format for token-budget accounting.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Classify this support ticket:\n" + bounded_text
                ),
            },
        ]

        estimated_input_tokens = sum(
            mock_llm.est_tokens(message["content"])
            for message in messages
        )

        # Never send a call whose estimated prompt exceeds the limit.
        if mock_llm.est_tokens(bounded_text) > MAX_INPUT_TOKENS:
            raise AssertionError(
                f"Ticket {ticket_id} exceeds the text token budget."
            )

        classification = classify_with_retry(bounded_text)

        attempts = int(classification["attempts"])
        total_calls += attempts

        # The deterministic mock returns usage for each successful call.
        # For budget comparison, report the estimated input and output cost
        # for all attempts, including failed attempts.
        estimated_output_tokens = mock_llm.est_tokens(
            json.dumps(
                {
                    "category": classification["category"],
                    "reason": classification["reason"],
                }
            )
        )

        ticket_input_tokens = estimated_input_tokens * attempts
        ticket_output_tokens = estimated_output_tokens * attempts

        total_input_tokens += ticket_input_tokens
        total_output_tokens += ticket_output_tokens

        if classification["status"] == "fallback":
            fallback_tickets += 1
            failed_tickets += 1

        results.append({
            "ticket_id": ticket_id,
            "category": classification["category"],
            "reason": classification["reason"],
            "status": classification["status"],
            "attempts": attempts,
            "pii_redacted": pii_changed,
            "input_truncated": truncated,
            "input_tokens": ticket_input_tokens,
            "output_tokens": ticket_output_tokens,
            "error": classification["error"],
        })

        budget_rows.append({
            "ticket_id": ticket_id,
            "input_tokens": ticket_input_tokens,
            "output_tokens": ticket_output_tokens,
            "total_tokens": (
                ticket_input_tokens + ticket_output_tokens
            ),
            "attempts": attempts,
            "status": classification["status"],
        })

    results_df = pd.DataFrame(results)
    budget_df = pd.DataFrame(budget_rows)

    results_df.to_csv(TRIAGE_PATH, index=False)
    budget_df.to_csv(BUDGET_PATH, index=False)

    category_counts = (
        results_df["category"].value_counts().to_dict()
    )

    summary = {
        "input_file": str(TICKETS_PATH),
        "ticket_count": int(len(tickets)),
        "output_ticket_count": int(len(results_df)),
        "category_counts": {
            str(key): int(value)
            for key, value in category_counts.items()
        },
        "pii_redacted_tickets": int(redacted_tickets),
        "truncated_tickets": int(truncated_tickets),
        "empty_tickets": int(empty_tickets),
        "fallback_tickets": int(fallback_tickets),
        "failed_tickets": int(failed_tickets),
        "model_calls_attempted": int(total_calls),
        "estimated_input_tokens": int(total_input_tokens),
        "estimated_output_tokens": int(total_output_tokens),
        "estimated_total_tokens": int(
            total_input_tokens + total_output_tokens
        ),
        "max_ticket_input_tokens": MAX_INPUT_TOKENS,
        "max_retries": MAX_RETRIES,
        "allowed_categories": sorted(ALLOWED_CATEGORIES),
        "mock_model_only": True,
        "network_calls": 0,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("PART C — SUPPORT TICKET TRIAGE COMPLETED")
    print("=" * 65)
    print(f"Tickets processed: {len(results_df)}")
    print(f"Category counts: {category_counts}")
    print(f"PII-redacted tickets: {redacted_tickets}")
    print(f"Truncated tickets: {truncated_tickets}")
    print(f"Empty tickets: {empty_tickets}")
    print(f"Fallback tickets: {fallback_tickets}")
    print(f"Estimated total tokens: {total_input_tokens + total_output_tokens}")
    print(f"Results: {TRIAGE_PATH}")
    print(f"Budget demo: {BUDGET_PATH}")
    print(f"Summary: {SUMMARY_PATH}")

    # Basic acceptance checks.
    assert len(results_df) == len(tickets)
    assert set(results_df["category"]).issubset(ALLOWED_CATEGORIES)
    assert results_df["ticket_id"].is_unique
    assert budget_df["ticket_id"].is_unique

    print("\n[PASS] All tickets have a triage result.")
    print("[PASS] All categories are allowed.")
    print("[PASS] No duplicate ticket IDs in outputs.")
    print("[PASS] Input text was redacted before classification.")


if __name__ == "__main__":
    main()
