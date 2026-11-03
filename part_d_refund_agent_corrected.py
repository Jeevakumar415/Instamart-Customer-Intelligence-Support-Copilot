"""Part D2: Safe, bounded, offline refund agent.

Uses mock_llm.py unchanged. Refunds are simulated only.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

import mock_llm
from part_d_rag import (
    build_retriever,
    load_policy_documents,
    retrieve_passages,
)

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"
SCENARIO_FILE = ROOT / "data" / "clean" / "agent_scenarios.csv"
POLICY_DIR = ROOT / "data" / "policies"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

RUN_LOG = OUTPUT_DIR / "agent_runs.jsonl"
REFUND_LEDGER = OUTPUT_DIR / "refund_ledger.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "agent_summary.json"
VALIDATION_FILE = OUTPUT_DIR / "agent_validation.json"
RESULTS_FILE = OUTPUT_DIR / "agent_scenario_results.csv"

MAX_AGENT_STEPS = 8
MAX_RETRIES = 3
CENT = Decimal("0.01")
APPROVAL_LIMIT = Decimal("1000.00")
MIN_DELIVERY_MINUTES = 45.0
VALID_STOP_REASONS = {
    "COMPLETED",
    "FALLBACK_MAX_ITERATIONS",
    "FALLBACK_TOKEN_LIMIT",
    "FALLBACK_TIMEOUT",
    "FALLBACK_MODEL_ERROR",
}

SYSTEM_PROMPT = (
    "AGENT\n"
    "Use tools to look up orders, search policies, calculate refunds, "
    "issue eligible simulated full refunds, or escalate refunds above INR 1000 "
    "for human approval. Only completed orders delivered in more than 45 minutes "
    "and requested within 7 days of delivery qualify. "
    "Never claim real money was transferred."
)

ORDER_RE = re.compile(r"\bORD-\d{6}\b", re.IGNORECASE)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_money(value: Any) -> Decimal | None:
    """Return a finite positive amount with cent precision, without rounding."""
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value).strip())
        if not amount.is_finite() or amount <= 0:
            return None
        cents = amount.quantize(CENT)
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        return None
    return cents if cents == amount else None


def _parse_delivery_minutes(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        minutes = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(minutes) or minutes < 0:
        return None
    return minutes


def _source_flag_is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return False


def _claims_refund_recorded(value: Any) -> bool:
    text = str(value or "").strip()
    return text.upper().startswith("REFUND:") or bool(
        re.search(r"\brefund\b.{0,80}\brecorded\b", text, flags=re.IGNORECASE | re.DOTALL)
    )


def _validate_terminal_tool_result(tool_name: str, result: Any) -> None:
    allowed_statuses = {
        "issue_refund": {"DENIED", "ESCALATE"},
        "escalate_to_human": {"NOT_FOUND", "DENIED", "ESCALATE"},
    }
    if tool_name not in allowed_statuses or not isinstance(result, dict):
        raise ValueError("Checkpoint contains an invalid terminal tool result.")

    status = result.get("status")
    reason = result.get("reason")
    if not isinstance(status, str) or status.upper() not in allowed_statuses[tool_name]:
        raise ValueError("Checkpoint contains an unsupported terminal status.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Checkpoint terminal result has no valid reason.")
    if result.get("real_money_moved") is not False:
        raise ValueError("Checkpoint terminal result must confirm no money moved.")
    if "refund_inr" in result and _parse_money(result.get("refund_inr")) is None:
        raise ValueError("Checkpoint terminal result has an invalid refund amount.")
    if _claims_refund_recorded(result.get("message")):
        raise ValueError("Checkpoint terminal result contains an unsupported refund claim.")


def _validate_checkpoint_trace(trace: Any) -> None:
    if not isinstance(trace, list) or any(not isinstance(item, dict) for item in trace):
        raise ValueError("Checkpoint trace contains invalid entries.")

    terminal_indexes = []
    for index, item in enumerate(trace):
        tool_name = item.get("tool")
        result = item.get("result")
        status = str(result.get("status", "")).upper() if isinstance(result, dict) else ""
        if status == "REFUND":
            raise ValueError("Checkpoint contains an unsupported REFUND result.")
        if status in {"ESCALATE", "DENIED", "NOT_FOUND"} and tool_name not in {
            "issue_refund", "escalate_to_human"
        }:
            raise ValueError("Checkpoint terminal status has an invalid tool name.")
        if tool_name in {"issue_refund", "escalate_to_human"}:
            _validate_terminal_tool_result(tool_name, result)
            terminal_indexes.append(index)

    if terminal_indexes and terminal_indexes != [len(trace) - 1]:
        raise ValueError("Checkpoint terminal tool result must be the final trace entry.")


def _validate_completed_checkpoint(checkpoint: dict[str, Any], result: Any) -> None:
    if not isinstance(result, dict):
        raise ValueError("Completed checkpoint has no valid final_result.")
    if result.get("run_id") != checkpoint.get("run_id"):
        raise ValueError("Completed checkpoint final_result has a mismatched run ID.")
    if result.get("real_money_moved") is not False:
        raise ValueError("Completed checkpoint must confirm no money moved.")
    final_response = result.get("final_response")
    if not isinstance(final_response, str):
        raise ValueError("Completed checkpoint has an invalid final_response.")
    if _claims_refund_recorded(final_response):
        raise ValueError("Completed checkpoint contains an unsupported refund claim.")

    trace = checkpoint.get("trace")
    _validate_checkpoint_trace(trace)
    if result.get("trace") != trace:
        raise ValueError("Completed checkpoint final_result trace does not match.")
    tool_steps = result.get("tool_steps")
    if (
        not isinstance(tool_steps, int)
        or isinstance(tool_steps, bool)
        or tool_steps != len(trace)
        or not isinstance(checkpoint.get("tool_steps"), int)
        or isinstance(checkpoint.get("tool_steps"), bool)
        or checkpoint.get("tool_steps") != tool_steps
    ):
        raise ValueError("Completed checkpoint tool_steps does not match its trace.")

    actual_outcome = result.get("actual_outcome")
    if not isinstance(actual_outcome, str) or actual_outcome not in {
        "ESCALATE", "DENIED", "NOT_FOUND", "FALLBACK"
    }:
        raise ValueError("Completed checkpoint has an unsupported actual_outcome.")
    if final_response != checkpoint.get("final_response"):
        raise ValueError("Completed checkpoint final_response does not match.")
    if checkpoint.get("actual_outcome") != actual_outcome:
        raise ValueError("Completed checkpoint actual_outcome does not match.")
    if result.get("stop_reason") not in VALID_STOP_REASONS:
        raise ValueError("Completed checkpoint has an invalid stop_reason.")
    if actual_outcome != RefundAgent.normalize_outcome(final_response, trace):
        raise ValueError("Completed checkpoint outcome does not match its result.")

    if trace and trace[-1].get("tool") in {"issue_refund", "escalate_to_human"}:
        terminal_status = str(trace[-1]["result"].get("status", "")).upper()
        if actual_outcome != terminal_status:
            raise ValueError("Completed checkpoint outcome does not match terminal tool status.")


def json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas scalars and other values to JSON-safe values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None

    if pd.isna(value):
        return None

    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=json_safe),
        encoding="utf-8",
    )


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(data, ensure_ascii=False, default=json_safe) + "\n"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    return records


def _read_refund_ledger_safely(
    path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read valid ledger rows and report corrupt rows without changing bytes."""
    if not path.exists():
        return [], []

    records: list[dict[str, Any]] = []
    errors: list[str] = []

    def reject_json_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(
                    raw_line.decode("utf-8"),
                    parse_float=Decimal,
                    parse_constant=reject_json_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"line_{line_number}:{type(exc).__name__}")
                continue

            if not isinstance(record, dict):
                errors.append(f"line_{line_number}:record_not_object")
                continue

            if record.get("status") == "SIMULATED_REFUND_RECORDED":
                order_id = record.get("order_id")
                amount = _parse_money(record.get("refund_inr"))
                if not isinstance(order_id, str) or not ORDER_RE.fullmatch(order_id):
                    errors.append(f"line_{line_number}:invalid_order_id")
                if amount is None:
                    errors.append(f"line_{line_number}:invalid_refund_amount")
                else:
                    record["refund_inr"] = amount
            elif not isinstance(record.get("status"), str):
                errors.append(f"line_{line_number}:invalid_status")

            records.append(record)

    return records, errors


def locate_column(
    columns: set[str],
    candidates: list[str],
) -> str | None:
    lookup = {name.lower(): name for name in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


class RefundAgent:
    """Offline refund agent with bounded execution and checkpoint support."""

    def __init__(
        self,
        ledger_path: str | Path | None = None,
        run_log_path: str | Path | None = None,
    ) -> None:
        self.refund_ledger_path = (
            Path(ledger_path) if ledger_path is not None else REFUND_LEDGER
        )
        self.run_log_path = (
            Path(run_log_path) if run_log_path is not None else RUN_LOG
        )
        for path in (DB_PATH, SCENARIO_FILE, POLICY_DIR):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required input missing: {path}"
                )

        self.db = sqlite3.connect(str(DB_PATH))
        self.db.row_factory = sqlite3.Row

        table_names = {
            row[0]
            for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

        if "orders" not in table_names:
            self.db.close()
            raise ValueError(
                "The SQLite database has no orders table."
            )

        columns = {
            row[1]
            for row in self.db.execute(
                'PRAGMA table_info("orders")'
            )
        }

        self.order_id_col = locate_column(
            columns, ["order_id", "id"]
        )
        self.status_col = locate_column(
            columns, ["status", "order_status"]
        )
        self.amount_col = locate_column(
            columns,
            ["order_amount", "amount", "total_amount", "total"],
        )
        self.delivery_col = locate_column(
            columns,
            [
                "delivery_minutes",
                "delivery_time_minutes",
                "delivery_time",
            ],
        )
        self.delivery_source_valid_col = locate_column(
            columns,
            ["delivery_minutes_source_valid"],
        )
        self.order_ts_col = locate_column(
            columns,
            ["order_ts", "order_timestamp", "created_at", "order_date"],
        )

        missing = [
            name
            for name, value in [
                ("order ID", self.order_id_col),
                ("status", self.status_col),
                ("amount", self.amount_col),
                ("delivery time", self.delivery_col),
            ]
            if value is None
        ]

        if missing:
            self.db.close()
            raise ValueError(
                f"Missing orders columns: {missing}. "
                f"Available: {sorted(columns)}"
            )

        self.passages = load_policy_documents()
        self.vectorizer, self.policy_matrix = build_retriever(
            self.passages
        )

        self.current_order_id: str | None = None
        # Request time is retained for audit/checkpoint context only; the
        # delivery date cannot be established from the available source data.
        self.current_request_ts: str | None = None

        # Preserve existing entries for duplicate/conflict review.
        self.ledger_records, self.ledger_errors = _read_refund_ledger_safely(
            self.refund_ledger_path
        )

    def close(self) -> None:
        self.db.close()

    def get_order(self, order_id: str) -> dict[str, Any]:
        order_id = str(order_id or "").strip().upper()

        if not ORDER_RE.fullmatch(order_id):
            return {"error": "invalid_order_id"}

        sql = (
            f'SELECT * FROM orders '
            f'WHERE "{self.order_id_col}" = ? LIMIT 1'
        )

        row = self.db.execute(sql, (order_id,)).fetchone()

        if row is None:
            return {
                "error": "order_not_found",
                "order_id": order_id,
            }

        raw = dict(row)
        status = str(
            raw.get(self.status_col, "") or ""
        ).strip().lower()

        amount = _parse_money(raw[self.amount_col])
        delivery = _parse_delivery_minutes(raw[self.delivery_col])
        delivery_source_valid = _source_flag_is_true(
            raw.get(self.delivery_source_valid_col)
            if self.delivery_source_valid_col
            else None
        )

        order_timestamp = None
        if self.order_ts_col:
            parsed_order_ts = pd.to_datetime(
                raw.get(self.order_ts_col), utc=True, errors="coerce"
            )
            if not pd.isna(parsed_order_ts):
                order_timestamp = parsed_order_ts.to_pydatetime()

        return {
            "order_id": order_id,
            "status": status,
            "order_amount": amount,
            "delivery_minutes": delivery,
            "delivery_minutes_source_valid": delivery_source_valid,
            "order_timestamp": (
                order_timestamp.isoformat() if order_timestamp else None
            ),
            # The source has no verified delivery timestamp. Do not derive one
            # from order_ts and delivery_minutes for refund policy decisions.
            "delivery_timestamp": None,
        }

    @staticmethod
    def _parse_utc_timestamp(value: Any) -> datetime | None:
        if value is None or str(value).strip() == "":
            return None
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()

    def _request_window_check(
        self,
        order: dict[str, Any],
        request_timestamp: Any = None,
    ) -> tuple[bool, str]:
        # Preserve the method signature for callers, but the source data does
        # not establish a delivery timestamp against which to validate the
        # policy's seven-day deadline.
        return False, "refund_deadline_unverifiable_from_order_data"

    def search_policy(self, query: str) -> dict[str, Any]:
        retrieved = retrieve_passages(
            str(query),
            self.passages,
            self.vectorizer,
            self.policy_matrix,
            top_k=3,
        )

        return {
            "query": str(query),
            "found": bool(retrieved),
            "passages": [
                {
                    "doc_name": item["doc_name"],
                    "passage_id": item["passage_id"],
                    "score": item["score"],
                    "text": item["text"],
                }
                for item in retrieved
            ],
        }

    def calculate_refund(
        self,
        order_amount: Any,
        delivery_minutes: Any,
        order_id: str | None = None,
    ) -> dict[str, Any]:
        """Calculate eligibility using trusted database values."""

        trusted_order_id = str(
            order_id or self.current_order_id or ""
        ).strip().upper()

        order = self.get_order(trusted_order_id)

        if "error" in order:
            return {
                "refund_inr": 0.0,
                "eligible": False,
                "reason": "order_not_found",
            }

        if order["status"] != "completed":
            return {
                "refund_inr": 0.0,
                "eligible": False,
                "reason": "order_not_completed",
            }

        # Do not trust amounts or delivery times supplied by the LLM.
        trusted_amount = _parse_money(order["order_amount"])
        trusted_delivery = _parse_delivery_minutes(order["delivery_minutes"])

        if trusted_amount is None:
            return {
                "refund_inr": 0.0,
                "eligible": False,
                "reason": "invalid_order_amount",
            }

        if (
            trusted_delivery is None
            or order.get("delivery_minutes_source_valid") is not True
        ):
            return {
                "order_id": trusted_order_id,
                "refund_inr": trusted_amount,
                "eligible": False,
                "approval_required": trusted_amount > APPROVAL_LIMIT,
                "requires_human_review": True,
                "reason": "delivery_duration_unverified_from_order_data",
            }

        if trusted_delivery <= MIN_DELIVERY_MINUTES:
            return {
                "refund_inr": 0.0,
                "eligible": False,
                "reason": "delivery_within_policy",
            }

        refund_amount = trusted_amount

        within_window, window_reason = self._request_window_check(order)
        if not within_window:
            return {
                "order_id": trusted_order_id,
                "refund_inr": refund_amount,
                "eligible": False,
                "approval_required": refund_amount > APPROVAL_LIMIT,
                "requires_human_review": True,
                "reason": window_reason,
            }
        return {
            "order_id": trusted_order_id,
            "refund_inr": refund_amount,
            "eligible": refund_amount > 0,
            "approval_required": refund_amount > APPROVAL_LIMIT,
            "reason": "late_delivery_full_refund",
        }


    def issue_refund(
        self,
        order_id: str,
        refund_inr: Any,
    ) -> dict[str, Any]:
        """Fail closed to human review; no real payment is ever made."""

        order_id = str(order_id or "").strip().upper()
        order = self.get_order(order_id)

        if "error" in order:
            return {
                "status": "DENIED",
                "reason": "order_not_found",
                "real_money_moved": False,
            }

        if order["status"] != "completed":
            return {
                "status": "DENIED",
                "reason": "only_completed_orders_are_eligible",
                "real_money_moved": False,
            }

        if order["order_amount"] is None:
            return {
                "status": "DENIED",
                "reason": "missing_order_amount",
                "real_money_moved": False,
            }

        expected = _parse_money(order["order_amount"])
        if expected is None:
            return {
                "status": "DENIED",
                "reason": "invalid_order_amount",
                "real_money_moved": False,
            }

        requested = _parse_money(refund_inr)
        if requested is None:
            return {
                "status": "DENIED",
                "reason": "invalid_refund_amount",
                "real_money_moved": False,
            }

        if requested != expected:
            return {
                "status": "DENIED",
                "reason": "refund_must_match_calculated_refund",
                "real_money_moved": False,
            }

        delivery_minutes = _parse_delivery_minutes(order["delivery_minutes"])
        if (
            delivery_minutes is None
            or order.get("delivery_minutes_source_valid") is not True
        ):
            return {
                "status": "ESCALATE",
                "order_id": order_id,
                "refund_inr": expected,
                "reason": "delivery_duration_unverified_from_order_data",
                "real_money_moved": False,
            }

        if delivery_minutes <= MIN_DELIVERY_MINUTES:
            return {
                "status": "DENIED",
                "reason": "delivery_did_not_exceed_45_minutes",
                "real_money_moved": False,
            }

        if self.ledger_errors:
            return {
                "status": "ESCALATE",
                "order_id": order_id,
                "refund_inr": expected,
                "reason": "refund_ledger_integrity_requires_human_review",
                "ledger_integrity_errors": list(self.ledger_errors),
                "real_money_moved": False,
            }

        existing_entries = [
            entry
            for entry in self.ledger_records
            if entry.get("order_id") == order_id
            and entry.get("status") == "SIMULATED_REFUND_RECORDED"
        ]
        if existing_entries:
            existing_amounts = [
                _parse_money(entry.get("refund_inr"))
                for entry in existing_entries
            ]
            if any(amount is None for amount in existing_amounts):
                return {
                    "status": "ESCALATE",
                    "order_id": order_id,
                    "refund_inr": expected,
                    "reason": "refund_ledger_integrity_requires_human_review",
                    "ledger_match": "invalid_existing_record",
                    "real_money_moved": False,
                }
            all_match = all(amount == expected for amount in existing_amounts)
            return {
                "status": "ESCALATE",
                "order_id": order_id,
                "refund_inr": expected,
                "reason": (
                    "existing_simulated_refund_requires_human_review"
                    if all_match
                    else "conflicting_existing_simulated_refund_requires_human_review"
                ),
                "ledger_match": (
                    "already_recorded"
                    if all_match
                    else "conflict"
                ),
                "real_money_moved": False,
            }

        return {
            "status": "ESCALATE",
            "order_id": order_id,
            "refund_inr": expected,
            "reason": (
                "refund_deadline_unverifiable_from_order_data"
                + (
                    ";human_approval_required_above_INR_1000"
                    if expected > APPROVAL_LIMIT
                    else ""
                )
            ),
            "real_money_moved": False,
        }

    def escalate_to_human(
        self,
        order_id: str,
        refund_inr: Any,
    ) -> dict[str, Any]:
        order_id = str(order_id or "").strip().upper()
        order = self.get_order(order_id)

        if "error" in order:
            result = {
                "status": "NOT_FOUND",
                "reason": "order_not_found",
                "real_money_moved": False,
            }

        elif order["status"] != "completed":
            result = {
                "status": "DENIED",
                "reason": "only_completed_orders_are_eligible",
                "real_money_moved": False,
            }

        elif order["order_amount"] is None:
            result = {
                "status": "DENIED",
                "reason": "missing_order_amount",
                "real_money_moved": False,
            }

        else:
            amount = _parse_money(order["order_amount"])
            if amount is None:
                result = {
                    "status": "DENIED",
                    "reason": "invalid_order_amount",
                    "real_money_moved": False,
                }
            else:
                delivery_minutes = _parse_delivery_minutes(
                    order["delivery_minutes"]
                )
                if (
                    delivery_minutes is None
                    or order.get("delivery_minutes_source_valid") is not True
                ):
                    reason = "delivery_duration_unverified_from_order_data"
                    result = {
                        "status": "ESCALATE",
                        "order_id": order_id,
                        "refund_inr": amount,
                        "reason": reason,
                        "real_money_moved": False,
                    }
                elif delivery_minutes <= MIN_DELIVERY_MINUTES:
                    result = {
                        "status": "DENIED",
                        "reason": "delivery_did_not_exceed_45_minutes",
                        "real_money_moved": False,
                    }
                else:
                    reason = "refund_deadline_unverifiable_from_order_data"
                    if amount > APPROVAL_LIMIT:
                        reason += ";human_approval_required_above_INR_1000"
                    result = {
                        "status": "ESCALATE",
                        "order_id": order_id,
                        "refund_inr": amount,
                        "reason": reason,
                        "real_money_moved": False,
                    }

        append_jsonl(
            self.run_log_path,
            {
                "timestamp": now_utc(),
                "event": "escalation_tool",
                "order_id": order_id,
                "result": result,
                "real_money_moved": False,
            },
        )

        return result

    def execute_tool(
        self,
        name: str,
        arguments: Any,
    ) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)

            except json.JSONDecodeError:
                return {
                    "error": "invalid_tool_arguments_json"
                }

        if not isinstance(arguments, dict):
            return {
                "error": "tool_arguments_must_be_an_object"
            }

        try:
            if name == "get_order":
                order_id = str(
                    arguments.get("order_id", "")
                ).strip().upper()

                if ORDER_RE.fullmatch(order_id):
                    self.current_order_id = order_id

                return self.get_order(order_id)

            if name == "search_policy":
                return self.search_policy(
                    arguments.get("query", "")
                )

            if name == "calculate_refund":
                return self.calculate_refund(
                    arguments.get("order_amount"),
                    arguments.get("delivery_minutes"),
                    arguments.get("order_id"),
                )

            if name == "issue_refund":
                return self.issue_refund(
                    arguments.get("order_id")
                    or self.current_order_id
                    or "",
                    arguments.get("refund_inr"),
                )

            if name == "escalate_to_human":
                return self.escalate_to_human(
                    arguments.get("order_id")
                    or self.current_order_id
                    or "",
                    arguments.get("refund_inr"),
                )

            return {
                "error": "unsupported_tool",
                "tool_name": name,
            }

        except Exception as exc:
            return {
                "error": "tool_execution_error",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }

    def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
    ) -> Any:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return mock_llm.chat(messages, tools=None)

            except mock_llm.TransientLLMError:
                if attempt == MAX_RETRIES:
                    raise

                time.sleep(0.05 * attempt)

        raise RuntimeError("Retry loop exited unexpectedly")

    @staticmethod
    def normalize_outcome(
        final_text: str,
        trace: list[dict[str, Any]],
    ) -> str:
        text = str(final_text or "").strip().upper()

        for label in (
            "ESCALATE",
            "DENIED",
            "NOT_FOUND",
            "FALLBACK",
        ):
            if (
                text.startswith(label + ":")
                or text.startswith(label + " ")
            ):
                return label

        for item in reversed(trace):
            result = item.get("result", {})
            status = str(result.get("status", "")).upper()

            if status in {
                "ESCALATE",
                "DENIED",
                "NOT_FOUND",
            }:
                return status

            if result.get("error") == "order_not_found":
                return "NOT_FOUND"

        if text.startswith("NEEDS_INFO"):
            return "FALLBACK"

        return "FALLBACK"

    @staticmethod
    def estimate_tokens(
        messages: list[dict[str, Any]],
    ) -> int:
        """Estimate tokens locally; mock_llm exposes no provider token usage."""
        encoded = json.dumps(
            messages,
            ensure_ascii=False,
            default=json_safe,
        )

        return max(1, (len(encoded) + 3) // 4)

    def save_checkpoint(
        self,
        run_id_or_checkpoint: str | dict[str, Any],
        state: dict[str, Any] | None = None,
        *,
        create_only: bool = False,
    ) -> Path:
        """Atomically save either the legacy state format or a D8 checkpoint."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        if isinstance(run_id_or_checkpoint, dict):
            checkpoint = dict(run_id_or_checkpoint)
            run_id = str(checkpoint.get("run_id", "")).strip()
            if not run_id:
                raise ValueError("Checkpoint requires a non-empty run_id.")
            checkpoint.setdefault("updated_at", now_utc())
            payload = checkpoint
        else:
            run_id = str(run_id_or_checkpoint).strip()
            if not run_id:
                raise ValueError("Checkpoint requires a non-empty run_id.")
            payload = {
                "run_id": run_id,
                "updated_at": now_utc(),
                "state": state if state is not None else {},
            }

        if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in run_id):
            raise ValueError("Invalid run_id; use letters, numbers, underscores, or hyphens.")

        checkpoint_path = STATE_DIR / f"{run_id}.json"
        serialized = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=json_safe,
            allow_nan=False,
        )
        if create_only and checkpoint_path.exists():
            raise FileExistsError(
                f"Checkpoint already exists for run_id={run_id!r}."
            )

        if isinstance(run_id_or_checkpoint, dict) and not create_only:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint does not exist for run_id={run_id!r}."
                )
            try:
                existing = json.loads(
                    checkpoint_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Existing checkpoint for run_id={run_id!r} is corrupt."
                ) from exc
            if not isinstance(existing, dict) or existing.get("run_id") != run_id:
                raise ValueError("Existing checkpoint has a mismatched run ID.")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{run_id}.",
            suffix=".tmp",
            dir=STATE_DIR,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            if create_only:
                try:
                    os.link(temporary_path, checkpoint_path)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Checkpoint already exists for run_id={run_id!r}."
                    ) from exc
            else:
                os.replace(temporary_path, checkpoint_path)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        return checkpoint_path

    def _continue_run_legacy(
        self,
        run_id: str,
        scenario: dict[str, Any],
        messages: list[dict[str, Any]],
        trace: list[dict[str, Any]],
        steps_used: int,
        retries_used: int,
        started_at: float,
    ) -> dict[str, Any]:
        final_text = ""
        error_message = None
        finished = False

        while steps_used < MAX_AGENT_STEPS and not finished:
            try:
                response = self.chat_with_retry(messages)

            except Exception as exc:
                error_message = (
                    f"{type(exc).__name__}: {exc}"
                )
                final_text = (
                    "FALLBACK: I could not complete the request "
                    "because the language model call failed."
                )
                break

            steps_used += 1

            if isinstance(response, str):
                final_text = response
                finished = True
                break

            if not isinstance(response, dict):
                final_text = (
                    "FALLBACK: The model returned an unsupported response."
                )
                finished = True
                break

            content = response.get("content")
            tool_call = response.get("tool_call")

            if not tool_call:
                final_text = str(content or "")
                finished = True
                break

            if not isinstance(tool_call, dict):
                final_text = (
                    "FALLBACK: The model returned an invalid tool call."
                )
                finished = True
                break

            tool_name = str(tool_call.get("name", ""))
            arguments = tool_call.get("arguments", {})

            result = self.execute_tool(tool_name, arguments)

            trace_item = {
                "step": steps_used,
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "timestamp": now_utc(),
            }
            trace.append(trace_item)

            append_jsonl(
                self.run_log_path,
                {
                    "timestamp": now_utc(),
                    "run_id": run_id,
                    "event": "tool_call",
                    **trace_item,
                },
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_call": tool_call,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                        default=json_safe,
                    ),
                }
            )

            self.save_checkpoint(
                run_id,
                {
                    "scenario": scenario,
                    "messages": messages,
                    "trace": trace,
                    "steps_used": steps_used,
                    "retries_used": retries_used,
                    "final_text": final_text,
                    "finished": finished,
                    "status": "running",
                },
            )

        if not finished and not error_message:
            final_text = (
                "FALLBACK: The maximum number of agent steps was reached."
            )

        outcome = self.normalize_outcome(final_text, trace)

        elapsed_seconds = round(time.monotonic() - started_at, 4)
        estimated_tokens = self.estimate_tokens(messages)

        result = {
            "run_id": run_id,
            "scenario_id": scenario.get("scenario_id"),
            "order_id": scenario.get("order_id"),
            "outcome": outcome,
            "final_text": final_text,
            "steps_used": steps_used,
            "retries_used": retries_used,
            "estimated_tokens": estimated_tokens,
            "elapsed_seconds": elapsed_seconds,
            "trace": trace,
            "error": error_message,
            "timestamp": now_utc(),
        }

        append_jsonl(
            self.run_log_path,
            {
                "timestamp": now_utc(),
                "run_id": run_id,
                "event": "run_completed",
                "outcome": outcome,
                "steps_used": steps_used,
                "estimated_tokens": estimated_tokens,
                "elapsed_seconds": elapsed_seconds,
                "error": error_message,
            },
        )

        self.save_checkpoint(
            run_id,
            {
                "scenario": scenario,
                "messages": messages,
                "trace": trace,
                "steps_used": steps_used,
                "retries_used": retries_used,
                "final_text": final_text,
                "finished": True,
                "status": "completed",
                "result": result,
            },
        )

        return result

    
   
    
    def _continue_run(
        self,
        checkpoint_or_run_id: dict[str, Any] | str,
        token_limit: int | dict[str, Any] = 2000,
        time_limit_seconds: float | list[dict[str, Any]] = 30.0,
        interrupt_after_tool_steps: int | int = 0,
        *legacy_args: Any,
    ) -> dict[str, Any]:
        """Continue a D8 checkpoint, while preserving the older D2 call signature."""
        # Legacy signature: (run_id, scenario, messages, trace, steps, retries, started_at)
        if isinstance(checkpoint_or_run_id, str):
            scenario = token_limit
            messages = time_limit_seconds
            trace = interrupt_after_tool_steps
            if len(legacy_args) < 3:
                raise TypeError("Invalid legacy _continue_run arguments.")
            steps_used, retries_used, started_at = legacy_args[:3]
            return self._continue_run_legacy(
                checkpoint_or_run_id,
                scenario,
                messages,
                trace,
                steps_used,
                retries_used,
                started_at,
            )

        checkpoint = checkpoint_or_run_id
        token_cap = int(token_limit) if isinstance(token_limit, (int, float)) else 2000
        time_cap = float(time_limit_seconds) if isinstance(time_limit_seconds, (int, float)) else 30.0
        interrupt_after = int(interrupt_after_tool_steps or 0)
        started = time.monotonic()
        deadline = started + max(0.0, time_cap)
        messages = checkpoint.setdefault("messages", [])
        trace = checkpoint.setdefault("trace", [])
        _validate_checkpoint_trace(trace)
        checkpoint["tool_steps"] = len(trace)
        checkpoint.setdefault("model_calls", 0)
        checkpoint.setdefault("estimated_tokens", self.estimate_tokens(messages))
        checkpoint.setdefault("iterations", 0)

        # If the final tool already ran before interruption, finalize from its
        # persisted result rather than asking the model to issue it again.
        if trace:
            last = trace[-1]
            last_result = last.get("result", {})
            last_tool = str(last.get("tool", ""))
            if last_tool in {"issue_refund", "escalate_to_human"} and isinstance(last_result, dict):
                status = str(last_result.get("status", "")).upper()
                reason = str(last_result.get("reason", "tool_result_received"))
                if status == "ESCALATE":
                    final_text = f"ESCALATE: human approval required. Reason: {reason}."
                elif status == "NOT_FOUND":
                    final_text = f"NOT_FOUND: {reason}."
                else:
                    final_text = f"DENIED: {reason}."
                return self._complete_checkpoint(
                    checkpoint, final_text, started, "COMPLETED"
                )

        final_text = ""
        error = ""
        stop_reason = "COMPLETED"
        while checkpoint["tool_steps"] < MAX_AGENT_STEPS:
            if time.monotonic() >= deadline:
                final_text = "FALLBACK: The agent time limit was reached."
                stop_reason = "FALLBACK_TIMEOUT"
                break
            if self.estimate_tokens(messages) >= token_cap:
                final_text = "FALLBACK: The agent token limit was reached."
                stop_reason = "FALLBACK_TOKEN_LIMIT"
                break

            try:
                response = self.chat_with_retry(messages)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                final_text = "FALLBACK: I could not complete the request because the language model call failed."
                stop_reason = "FALLBACK_MODEL_ERROR"
                break

            checkpoint["model_calls"] = int(checkpoint.get("model_calls", 0)) + 1
            checkpoint["iterations"] = int(checkpoint.get("iterations", 0)) + 1

            # A response that arrives after the time budget expires must not
            # trigger additional parsing or tool execution.
            if time.monotonic() >= deadline:
                final_text = "FALLBACK: The agent time limit was reached."
                stop_reason = "FALLBACK_TIMEOUT"
                break

            if isinstance(response, str):
                final_text = response
                if self.normalize_outcome(final_text, trace) == "FALLBACK":
                    stop_reason = "FALLBACK_MODEL_ERROR"
                break

            # mock_llm mirrors the hosted chat-completions API and returns
            # response.choices[0].message, not a plain dictionary.
            content = None
            tool_call = None
            if isinstance(response, dict):
                content = response.get("content")
                tool_call = response.get("tool_call")
            else:
                choices = getattr(response, "choices", None)
                choice = choices[0] if choices else None
                message = getattr(choice, "message", None) if choice else None
                if message is None:
                    final_text = "FALLBACK: The model returned an unsupported response."
                    stop_reason = "FALLBACK_MODEL_ERROR"
                    break
                content = getattr(message, "content", None)
                tool_calls = getattr(message, "tool_calls", None) or []
                if tool_calls:
                    raw_call = tool_calls[0]
                    raw_name = getattr(raw_call, "name", None)
                    raw_arguments = getattr(raw_call, "arguments", {})
                    if isinstance(raw_arguments, str):
                        try:
                            raw_arguments = json.loads(raw_arguments)
                        except json.JSONDecodeError:
                            final_text = "FALLBACK: The model returned invalid tool arguments."
                            stop_reason = "FALLBACK_MODEL_ERROR"
                            break
                    tool_call = {
                        "id": getattr(raw_call, "id", ""),
                        "name": raw_name,
                        "arguments": raw_arguments,
                    }

            if not tool_call:
                final_text = str(content or "")
                break
            if not isinstance(tool_call, dict):
                final_text = "FALLBACK: The model returned an invalid tool call."
                stop_reason = "FALLBACK_MODEL_ERROR"
                break

            tool_name = str(tool_call.get("name", ""))
            arguments = tool_call.get("arguments", {})
            result = self.execute_tool(tool_name, arguments)
            if tool_name in {"issue_refund", "escalate_to_human"}:
                _validate_terminal_tool_result(tool_name, result)
            trace_item = {
                "step": len(trace) + 1,
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "timestamp": now_utc(),
            }
            trace.append(trace_item)
            checkpoint["tool_steps"] = len(trace)
            checkpoint["current_order_id"] = self.current_order_id
            checkpoint["request_ts"] = self.current_request_ts
            messages.append({"role": "assistant", "content": content or "", "tool_call": tool_call})
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False, default=json_safe),
            })
            checkpoint["estimated_tokens"] = self.estimate_tokens(messages)
            checkpoint["status"] = "RUNNING"
            self.save_checkpoint(checkpoint)
            append_jsonl(self.run_log_path, {
                "timestamp": now_utc(),
                "run_id": checkpoint["run_id"],
                "event": "tool_call",
                **trace_item,
            })

            if interrupt_after and checkpoint["tool_steps"] >= interrupt_after:
                checkpoint["status"] = "INTERRUPTED_FOR_RESUME_DEMO"
                self.save_checkpoint(checkpoint)
                return {
                    "run_id": checkpoint["run_id"],
                    "scenario_id": checkpoint.get("scenario_id"),
                    "status": "INTERRUPTED_FOR_RESUME_DEMO",
                    "actual_outcome": "INTERRUPTED",
                    "final_response": "INTERRUPTED_FOR_RESUME_DEMO",
                    "tool_steps": checkpoint["tool_steps"],
                    "trace": trace,
                    "error": "",
                    "stop_reason": "INTERRUPTED_FOR_RESUME_DEMO",
                    "model_calls": checkpoint["model_calls"],
                    "estimated_tokens": checkpoint["estimated_tokens"],
                    "elapsed_seconds": round(time.monotonic() - started, 4),
                    "real_money_moved": False,
                }

            # The tool result is terminal; don't invoke the model again on resume.
            if tool_name in {"issue_refund", "escalate_to_human"}:
                status = str(result.get("status", "")).upper() if isinstance(result, dict) else ""
                if status in {"ESCALATE", "DENIED", "NOT_FOUND"}:
                    if status == "ESCALATE":
                        final_text = f"ESCALATE: human approval required. Reason: {result.get('reason', 'tool_result_received')}."
                    elif status == "NOT_FOUND":
                        final_text = f"NOT_FOUND: {result.get('reason', 'order_not_found')}."
                    else:
                        final_text = f"DENIED: {result.get('reason', 'refund_denied')}."
                    break

        if not final_text:
            final_text = "FALLBACK: The maximum number of agent steps was reached."
            stop_reason = "FALLBACK_MAX_ITERATIONS"
        elif (
            stop_reason == "COMPLETED"
            and self.normalize_outcome(final_text, trace) == "FALLBACK"
        ):
            # A fallback outcome cannot be reported as successful completion,
            # even if a model returns fallback text without raising.
            stop_reason = "FALLBACK_MODEL_ERROR"
        return self._complete_checkpoint(
            checkpoint, final_text, started, stop_reason, error
        )

    def _complete_checkpoint(
        self,
        checkpoint: dict[str, Any],
        final_text: str,
        started: float,
        stop_reason: str,
        error: str = "",
    ) -> dict[str, Any]:
        _validate_checkpoint_trace(checkpoint.get("trace", []))
        if _claims_refund_recorded(final_text):
            final_text = "FALLBACK: I cannot confirm the requested refund result."
            if stop_reason == "COMPLETED":
                stop_reason = "FALLBACK_MODEL_ERROR"

        trace = checkpoint.get("trace", [])
        outcome = self.normalize_outcome(final_text, trace)
        result = {
            "timestamp": now_utc(),
            "run_id": checkpoint["run_id"],
            "scenario_id": checkpoint.get("scenario_id"),
            "customer_id": checkpoint.get("customer_id"),
            "order_id": checkpoint.get("order_id"),
            "message": checkpoint.get("message", ""),
            "expected_outcome": checkpoint.get("expected_outcome", ""),
            "actual_outcome": outcome,
            "expected_refund_inr": checkpoint.get("expected_refund_inr"),
            "final_response": final_text,
            "tool_steps": len(trace),
            "trace": trace,
            "error": error,
            "stop_reason": stop_reason,
            "model_calls": int(checkpoint.get("model_calls", 0)),
            "estimated_tokens": self.estimate_tokens(checkpoint.get("messages", [])),
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "real_money_moved": False,
        }
        checkpoint.update({
            "status": "COMPLETED",
            "final_result": result,
            "final_response": final_text,
            "actual_outcome": outcome,
            "tool_steps": len(trace),
        })
        _validate_completed_checkpoint(checkpoint, result)
        self.save_checkpoint(checkpoint)
        append_jsonl(self.run_log_path, {
            "timestamp": now_utc(),
            "run_id": checkpoint["run_id"],
            "scenario_id": checkpoint.get("scenario_id"),
            "event": "scenario_completed",
            "expected_outcome": result["expected_outcome"],
            "actual_outcome": outcome,
            "tool_steps": result["tool_steps"],
            "model_calls": result["model_calls"],
            "estimated_tokens": result["estimated_tokens"],
            "stop_reason": stop_reason,
            "error": error,
            "real_money_moved": False,
        })
        return result

    def resume_agent(
        self,
        run_id: str,
        token_limit: int = 2000,
        time_limit_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Resume an interrupted run from its latest saved checkpoint."""
        run_id = str(run_id or "").strip()
        if not run_id or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for ch in run_id
        ):
            raise ValueError("Invalid run ID.")
        path = STATE_DIR / f"{run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint found for run_id={run_id!r}")
        try:
            checkpoint = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"Non-finite JSON constant: {value}")
                ),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Checkpoint for run_id={run_id!r} is unreadable or corrupt."
            ) from exc
        if not isinstance(checkpoint, dict) or checkpoint.get("run_id") != run_id:
            raise ValueError("Invalid checkpoint or mismatched run ID.")
        if checkpoint.get("status") == "COMPLETED":
            result = checkpoint.get("final_result")
            _validate_completed_checkpoint(checkpoint, result)
            return result
        if checkpoint.get("status") not in {"RUNNING", "INTERRUPTED_FOR_RESUME_DEMO"}:
            raise ValueError(f"Checkpoint cannot be resumed from status {checkpoint.get('status')!r}.")
        if not isinstance(checkpoint.get("messages"), list) or not isinstance(checkpoint.get("trace"), list):
            raise ValueError("Checkpoint messages or trace are invalid.")
        if any(not isinstance(item, dict) for item in checkpoint["messages"]):
            raise ValueError("Checkpoint messages contain invalid entries.")
        if any(not isinstance(item, dict) for item in checkpoint["trace"]):
            raise ValueError("Checkpoint trace contains invalid entries.")
        _validate_checkpoint_trace(checkpoint["trace"])
        tool_steps = checkpoint.get("tool_steps")
        if (
            not isinstance(tool_steps, int)
            or isinstance(tool_steps, bool)
            or tool_steps != len(checkpoint["trace"])
        ):
            raise ValueError("Checkpoint tool_steps does not match trace length.")
        if checkpoint.get("current_order_id") is not None and not isinstance(
            checkpoint.get("current_order_id"), str
        ):
            raise ValueError("Checkpoint current_order_id is invalid.")
        if checkpoint.get("request_ts") is not None and not isinstance(
            checkpoint.get("request_ts"), str
        ):
            raise ValueError("Checkpoint request_ts is invalid.")
        self.current_order_id = checkpoint.get("current_order_id")
        self.current_request_ts = checkpoint.get("request_ts")
        checkpoint["status"] = "RUNNING"
        self.save_checkpoint(checkpoint)
        return self._continue_run(checkpoint, token_limit, time_limit_seconds)

    def run_agent(
        self,
        scenario: dict[str, Any],
        token_limit: int = 2000,
        time_limit_seconds: float = 30.0,
        run_id: str | None = None,
        interrupt_after_tool_steps: int | None = None,
    ) -> dict[str, Any]:
        scenario_id = str(scenario["scenario_id"])
        customer_id = str(scenario["customer_id"])
        message_text = str(scenario["message"])

        expected_outcome = str(
            scenario["expected_outcome"]
        ).strip().upper()

        match = ORDER_RE.search(message_text)
        order_id = match.group(0).upper() if match else None

        # Retain an explicitly supplied request time for checkpoint context.
        # It cannot establish the seven-day rule without a verified delivery
        # timestamp, so it is never used to approve a refund.
        self.current_request_ts = str(scenario.get("request_ts") or now_utc())

        user_prompt = (
            f"SCENARIO_ID: {scenario_id}\n"
            f"CUSTOMER_ID: {customer_id}\n"
            f"MESSAGE: {message_text}"
        )

        checkpoint = {
            "run_id": run_id
            or f"run_{scenario_id}_{int(time.time() * 1000)}",
            "status": "RUNNING",
            "scenario_id": scenario_id,
            "customer_id": customer_id,
            "order_id": order_id,
            "message": message_text,
            "expected_outcome": expected_outcome,
            "expected_refund_inr": scenario.get(
                "expected_refund_inr"
            ),
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "trace": [],
            "tool_steps": 0,
            "model_calls": 0,
            "estimated_tokens": 0,
            "iterations": 0,
            "current_order_id": None,
            "request_ts": self.current_request_ts,
        }

        self.save_checkpoint(checkpoint, create_only=True)

        return self._continue_run(
            checkpoint,
            token_limit,
            time_limit_seconds,
            interrupt_after_tool_steps,
        )
