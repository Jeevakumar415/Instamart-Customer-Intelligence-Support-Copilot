"""Independent D7 fallback tests for part_d_refund_agent.py.

Run from the project root with:
    python test_part_d7_fallbacks.py

All generated checkpoints, logs, and refund ledger entries are isolated in a
temporary directory. The harness does not edit mock_llm.py or generate_data.py.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import mock_llm
import part_d_refund_agent_corrected as agent_module
from part_d_refund_agent import RefundAgent


def print_result(label: str, result: dict[str, Any], expected_stop: str) -> bool:
    actual_stop = str(result.get("stop_reason", ""))
    actual_outcome = str(result.get("actual_outcome", ""))
    passed = actual_stop == expected_stop and actual_outcome == "FALLBACK"

    print(f"\n[{label}] {'PASS' if passed else 'FAIL'}")
    print(f"Expected stop reason: {expected_stop}")
    print(f"Actual stop reason:   {actual_stop}")
    print(f"Actual outcome:       {actual_outcome}")
    print(f"Tool steps:           {result.get('tool_steps')}")
    print(f"Model calls:          {result.get('model_calls')}")
    print(f"Estimated tokens:     {result.get('estimated_tokens')}")
    print(f"Error:                {result.get('error')}")
    print(f"Final response:       {result.get('final_response')}")
    return passed


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="instamart-d7-") as temp_dir:
        isolated_root = Path(temp_dir)
        isolated_state = isolated_root / "state"
        isolated_outputs = isolated_root / "outputs"
        isolated_state.mkdir()
        isolated_outputs.mkdir()
        with patch.multiple(
            agent_module,
            STATE_DIR=isolated_state,
            RUN_LOG=isolated_outputs / "agent_runs.jsonl",
            REFUND_LEDGER=isolated_outputs / "refund_ledger.jsonl",
            SUMMARY_FILE=isolated_outputs / "agent_summary.json",
            VALIDATION_FILE=isolated_outputs / "agent_validation.json",
            RESULTS_FILE=isolated_outputs / "agent_scenario_results.csv",
        ):
            return _run_isolated_cases()


def _run_isolated_cases() -> int:
    stamp = int(time.time() * 1000)
    agent = RefundAgent()
    results: list[bool] = []

    loop_scenario = {
        "scenario_id": "D7_MAX_ITERATIONS_TEST",
        "customer_id": "D7_TEST",
        "message": "[LOOP] Refund order ORD-100048.",
        "expected_outcome": "FALLBACK",
        "expected_refund_inr": 0,
    }

    try:
        # 1) High token budget and normal time budget isolate the iteration cap.
        if hasattr(mock_llm, "reset"):
            mock_llm.reset()
        max_result = agent.run_agent(
            loop_scenario,
            token_limit=100_000,
            time_limit_seconds=30.0,
            run_id=f"D7_MAX_ITERATIONS_{stamp}",
        )
        results.append(
            print_result(
                "MAX ITERATIONS",
                max_result,
                "FALLBACK_MAX_ITERATIONS",
            )
        )

        # 2) A deliberately tiny token budget should stop before a model call.
        token_result = agent.run_agent(
            {
                "scenario_id": "D7_TOKEN_LIMIT_TEST",
                "customer_id": "D7_TEST",
                "message": "Please check refund eligibility for ORD-100048.",
                "expected_outcome": "FALLBACK",
                "expected_refund_inr": 0,
            },
            token_limit=1,
            time_limit_seconds=30.0,
            run_id=f"D7_TOKEN_LIMIT_{stamp}",
        )
        results.append(
            print_result(
                "TOKEN LIMIT",
                token_result,
                "FALLBACK_TOKEN_LIMIT",
            )
        )

        # 3) A zero-second time budget should trigger timeout before model call.
        timeout_result = agent.run_agent(
            {
                "scenario_id": "D7_TIMEOUT_TEST",
                "customer_id": "D7_TEST",
                "message": "Please check refund eligibility for ORD-100048.",
                "expected_outcome": "FALLBACK",
                "expected_refund_inr": 0,
            },
            token_limit=100_000,
            time_limit_seconds=0.0,
            run_id=f"D7_TIMEOUT_{stamp}",
        )
        results.append(
            print_result(
                "TIMEOUT",
                timeout_result,
                "FALLBACK_TIMEOUT",
            )
        )

        zero_timeout_stopped_before_model = (
            timeout_result.get("model_calls") == 0
            and timeout_result.get("tool_steps") == 0
        )
        print(
            "[TIMEOUT PRE-CALL CHECK] "
            f"{'PASS' if zero_timeout_stopped_before_model else 'FAIL'}"
        )
        results.append(zero_timeout_stopped_before_model)

        def fail_model(_messages: list[dict[str, Any]]) -> Any:
            raise RuntimeError("forced D7 model failure")

        with patch.object(agent, "chat_with_retry", side_effect=fail_model):
            model_error_result = agent.run_agent(
                {
                    "scenario_id": "D7_MODEL_ERROR_TEST",
                    "customer_id": "D7_TEST",
                    "message": "Please check refund eligibility for ORD-100048.",
                    "expected_outcome": "FALLBACK",
                    "expected_refund_inr": 0,
                },
                token_limit=100_000,
                time_limit_seconds=30.0,
                run_id=f"D7_MODEL_ERROR_{stamp}",
            )
        results.append(
            print_result(
                "MODEL ERROR",
                model_error_result,
                "FALLBACK_MODEL_ERROR",
            )
        )

        ledger_empty = not agent_module.read_jsonl(agent_module.REFUND_LEDGER)
        print(f"[NO SIMULATED REFUNDS] {'PASS' if ledger_empty else 'FAIL'}")
        results.append(ledger_empty)

        summary = {
            "component": "Part D7 - Independent fallback tests",
            "passed": sum(results),
            "failed": len(results) - sum(results),
            "tests": [
                {
                    "test": "max_iterations",
                    "stop_reason": max_result.get("stop_reason"),
                    "outcome": max_result.get("actual_outcome"),
                    "tool_steps": max_result.get("tool_steps"),
                    "error": max_result.get("error"),
                },
                {
                    "test": "token_limit",
                    "stop_reason": token_result.get("stop_reason"),
                    "outcome": token_result.get("actual_outcome"),
                    "tool_steps": token_result.get("tool_steps"),
                    "error": token_result.get("error"),
                },
                {
                    "test": "timeout",
                    "stop_reason": timeout_result.get("stop_reason"),
                    "outcome": timeout_result.get("actual_outcome"),
                    "tool_steps": timeout_result.get("tool_steps"),
                    "error": timeout_result.get("error"),
                },
                {
                    "test": "model_error",
                    "stop_reason": model_error_result.get("stop_reason"),
                    "outcome": model_error_result.get("actual_outcome"),
                    "tool_steps": model_error_result.get("tool_steps"),
                    "error": model_error_result.get("error"),
                },
            ],
        }
        print("\n" + "=" * 60)
        print("D7 FALLBACK TEST SUMMARY")
        print("=" * 60)
        print(json.dumps(summary, indent=2))
        return 0 if summary["failed"] == 0 else 1
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
