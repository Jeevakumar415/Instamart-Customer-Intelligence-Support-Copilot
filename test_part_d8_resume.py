"""Part D8: checkpoint/resume verification."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import mock_llm
import part_d_refund_agent_corrected as agent_module
import part_d_rag as rag_module
from part_d_refund_agent import RefundAgent


RUN_ID = "D8_CHECKPOINT_RESUME_DEMO"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="instamart-d8-") as temp_dir:
        isolated_root = Path(temp_dir)
        isolated_state = isolated_root / "state"
        isolated_outputs = isolated_root / "outputs"
        isolated_inputs = isolated_root / "inputs"
        isolated_policies = isolated_inputs / "policies"
        isolated_state.mkdir()
        isolated_outputs.mkdir()
        isolated_policies.mkdir(parents=True)
        scenario_file = isolated_inputs / "agent_scenarios.csv"
        scenario_file.write_text("scenario_id\n", encoding="utf-8")
        (isolated_policies / "refund_policy.txt").write_text(
            "Refund requests within seven days require a verified delivery date.",
            encoding="utf-8",
        )
        database_path = isolated_inputs / "instamart.db"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute(
                "CREATE TABLE orders (order_id TEXT, status TEXT, order_amount REAL, "
                "delivery_minutes REAL, delivery_minutes_source_valid INTEGER, order_ts TEXT)"
            )
            connection.execute(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
                ("ORD-100048", "completed", 128.25, 65.0, 1, "2025-01-01 10:00:00"),
            )
            connection.commit()
        finally:
            connection.close()
        with patch.multiple(
            agent_module,
            STATE_DIR=isolated_state,
            DB_PATH=database_path,
            SCENARIO_FILE=scenario_file,
            POLICY_DIR=isolated_policies,
            RUN_LOG=isolated_outputs / "agent_runs.jsonl",
            REFUND_LEDGER=isolated_outputs / "refund_ledger.jsonl",
            SUMMARY_FILE=isolated_outputs / "agent_summary.json",
            VALIDATION_FILE=isolated_outputs / "agent_validation.json",
            RESULTS_FILE=isolated_outputs / "agent_scenario_results.csv",
        ):
            with patch.object(rag_module, "POLICY_DIR", isolated_policies):
                return _run_isolated_resume(isolated_state, isolated_outputs)


def _run_isolated_resume(
    isolated_state: Path,
    isolated_outputs: Path,
) -> int:
    # Use a fresh run ID each time to avoid stale checkpoints.
    import time
    run_id = f"{RUN_ID}_{int(time.time() * 1000)}"

    scenario = {
        "scenario_id": run_id,
        "customer_id": "D8_TEST",
        "message": (
            "My order ORD-100048 reached me very late. "
            "I would like a refund."
        ),
        "expected_outcome": "ESCALATE",
        "expected_refund_inr": 128.25,
    }

    mock_llm.reset()
    agent = RefundAgent()

    try:
        print("=" * 60)
        print("PART D8 — CHECKPOINT / RESUME TEST")
        print("=" * 60)

        interrupted = agent.run_agent(
            scenario,
            token_limit=100000,
            time_limit_seconds=30.0,
            run_id=run_id,
            interrupt_after_tool_steps=4,
        )

        checkpoint_path = isolated_state / f"{run_id}.json"
        # A neighboring checkpoint is evidence that resume only updates its
        # own run ID.
        sibling_path = isolated_state / f"{run_id}_SIBLING.json"
        sibling_bytes = json.dumps(
            {"run_id": f"{run_id}_SIBLING", "status": "SENTINEL"}
        ).encode("utf-8")
        sibling_path.write_bytes(sibling_bytes)
        saved_before = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )

        trace_before = saved_before.get("trace", [])
        steps_before = [
            item.get("step") for item in trace_before
        ]
        tools_before = [
            item.get("tool") for item in trace_before
        ]

        print("\n[INTERRUPTION]")
        print(f"Status: {interrupted.get('status')}")
        print(f"Completed tools: {len(trace_before)}")

        resumed = agent.resume_agent(
            run_id,
            token_limit=100000,
            time_limit_seconds=30.0,
        )

        saved_after = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        sibling_preserved = sibling_path.read_bytes() == sibling_bytes
        trace_after = saved_after.get("trace", [])

        steps_after = [
            item.get("step") for item in trace_after
        ]
        tools_after = [
            item.get("tool") for item in trace_after
        ]
        issue_refund_results = [
            item.get("result", {})
            for item in trace_after
            if item.get("tool") == "issue_refund"
        ]
        issue_refund_result = (
            issue_refund_results[-1] if issue_refund_results else {}
        )
        request_ts_before = saved_before.get("request_ts")
        request_ts_after = saved_after.get("request_ts")
        ledger_empty = not agent_module.read_jsonl(agent_module.REFUND_LEDGER)

        checks = {
            "interrupted_after_four_tools": (
                interrupted.get("status")
                == "INTERRUPTED_FOR_RESUME_DEMO"
                and len(trace_before) == 4
            ),
            "checkpoint_preserved": (
                steps_after[:4] == steps_before
                and tools_after[:4] == tools_before
            ),
            "resume_updates_only_intended_checkpoint": sibling_preserved,
            "no_duplicate_tool_steps": (
                len(steps_after) == len(set(steps_after))
            ),
            "exactly_four_tools": len(trace_after) == 4,
            "run_completed": (
                saved_after.get("status") == "COMPLETED"
                and resumed.get("stop_reason") == "COMPLETED"
            ),
            "escalation_outcome": (
                resumed.get("actual_outcome") == "ESCALATE"
                and issue_refund_result.get("reason")
                == "refund_deadline_unverifiable_from_order_data"
            ),
            "order_id_preserved": (
                saved_after.get("current_order_id") == "ORD-100048"
            ),
            "request_context_preserved": (
                bool(request_ts_before)
                and request_ts_after == request_ts_before
            ),
            "exactly_one_issue_refund_call": len(issue_refund_results) == 1,
            "no_refund_ledger_entry": ledger_empty,
            "simulation_only": (
                resumed.get("real_money_moved") is False
            ),
            "terminal_tool_result_reused": (
                len(issue_refund_results) == 1
                and issue_refund_results[0].get("status") == "ESCALATE"
                and saved_after.get("tool_steps") == saved_before.get("tool_steps")
            ),
        }

        print("\n[RESUME RESULT]")
        print(f"Status: {saved_after.get('status')}")
        print(f"Outcome: {resumed.get('actual_outcome')}")
        print(f"Tool steps: {resumed.get('tool_steps')}")
        print(f"Final response: {resumed.get('final_response')}")

        print("\n[VALIDATION]")
        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")

        passed = sum(checks.values())
        failed = len(checks) - passed

        summary = {
            "component": "Part D8 - Checkpoint and resume",
            "run_id": run_id,
            "passed": passed,
            "failed": failed,
            "checks": checks,
            "tools_before_resume": tools_before,
            "tools_after_resume": tools_after,
        }

        output_path = isolated_outputs / "d8_resume_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print(f"\nSummary saved in isolated temp directory: {output_path}")
        print(f"D8 validation totals: {passed} passed, {failed} failed")

        # A reused run ID must fail before changing existing checkpoint bytes.
        collision_bytes = checkpoint_path.read_bytes()
        try:
            agent.run_agent(
                scenario,
                token_limit=100000,
                time_limit_seconds=30.0,
                run_id=run_id,
            )
            checks["run_id_collision_fails_closed"] = False
        except FileExistsError:
            checks["run_id_collision_fails_closed"] = (
                checkpoint_path.read_bytes() == collision_bytes
            )

        # Corrupt checkpoint data must be rejected without modifying it or
        # making another model call.
        corrupt_id = f"{run_id}_CORRUPT"
        corrupt_path = isolated_state / f"{corrupt_id}.json"
        corrupt_bytes = b'{"run_id":'
        corrupt_path.write_bytes(corrupt_bytes)
        with patch.object(agent, "chat_with_retry") as model_call:
            try:
                agent.resume_agent(corrupt_id)
                checks["corrupt_checkpoint_rejected_without_model_call"] = False
            except ValueError:
                checks["corrupt_checkpoint_rejected_without_model_call"] = (
                    corrupt_path.read_bytes() == corrupt_bytes
                    and not model_call.called
                )

        valid_escalation_trace = [{
            "step": 1,
            "tool": "issue_refund",
            "arguments": {"order_id": "ORD-100048", "refund_inr": 128.25},
            "result": {
                "status": "ESCALATE",
                "order_id": "ORD-100048",
                "refund_inr": 128.25,
                "reason": "refund_deadline_unverifiable_from_order_data",
                "real_money_moved": False,
            },
            "timestamp": "2026-01-01T00:00:00+00:00",
        }]

        def completed_payload(
            case_id: str,
            trace: list[dict[str, object]],
            outcome: str,
            response: str,
        ) -> dict[str, object]:
            final_result = {
                "run_id": case_id,
                "actual_outcome": outcome,
                "final_response": response,
                "trace": trace,
                "tool_steps": len(trace),
                "stop_reason": "COMPLETED",
                "real_money_moved": False,
            }
            return {
                "run_id": case_id,
                "status": "COMPLETED",
                "trace": trace,
                "tool_steps": len(trace),
                "actual_outcome": outcome,
                "final_response": response,
                "final_result": final_result,
            }

        def assert_rejected_without_rewrite(case_id: str, payload: dict[str, object]) -> bool:
            path = isolated_state / f"{case_id}.json"
            original = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            path.write_bytes(original)
            with patch.object(agent, "chat_with_retry") as model_call:
                try:
                    agent.resume_agent(case_id)
                    return False
                except ValueError:
                    return path.read_bytes() == original and not model_call.called

        forged_id = f"{run_id}_FORGED_REFUND"
        forged_refund_trace = [{
            **valid_escalation_trace[0],
            "result": {
                "status": "REFUND",
                "order_id": "ORD-100048",
                "refund_inr": 128.25,
                "reason": "legacy result",
                "real_money_moved": False,
            },
        }]
        checks["forged_completed_refund_rejected_unchanged"] = (
            assert_rejected_without_rewrite(
                forged_id,
                completed_payload(
                    forged_id,
                    forged_refund_trace,
                    "REFUND",
                    "REFUND: simulated refund recorded.",
                ),
            )
        )

        running_refund_id = f"{run_id}_RUNNING_REFUND"
        running_refund_payload = {
            "run_id": running_refund_id,
            "status": "RUNNING",
            "messages": [],
            "trace": forged_refund_trace,
            "tool_steps": 1,
            "current_order_id": "ORD-100048",
            "request_ts": "2026-01-01T00:00:00+00:00",
        }
        checks["running_refund_trace_rejected_unchanged"] = (
            assert_rejected_without_rewrite(running_refund_id, running_refund_payload)
        )

        mismatch_id = f"{run_id}_MISMATCHED_OUTCOME"
        checks["mismatched_outcome_rejected_unchanged"] = (
            assert_rejected_without_rewrite(
                mismatch_id,
                completed_payload(
                    mismatch_id,
                    valid_escalation_trace,
                    "DENIED",
                    "ESCALATE: human review required.",
                ),
            )
        )

        malformed_id = f"{run_id}_MALFORMED_TERMINAL"
        malformed_trace = [{
            **valid_escalation_trace[0],
            "result": {
                "status": "ESCALATE",
                "refund_inr": 128.25,
                "real_money_moved": False,
            },
        }]
        checks["malformed_terminal_result_rejected_unchanged"] = (
            assert_rejected_without_rewrite(
                malformed_id,
                {
                    "run_id": malformed_id,
                    "status": "RUNNING",
                    "messages": [],
                    "trace": malformed_trace,
                    "tool_steps": 1,
                },
            )
        )

        checks["no_ledger_entries_after_invalid_checkpoint_rejections"] = (
            not agent_module.REFUND_LEDGER.exists()
            or not agent_module.read_jsonl(agent_module.REFUND_LEDGER)
        )

        passed = sum(checks.values())
        failed = len(checks) - passed
        summary["passed"] = passed
        summary["failed"] = failed
        summary["checks"] = checks
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for name in (
            "run_id_collision_fails_closed",
            "corrupt_checkpoint_rejected_without_model_call",
        ):
            print(f"[{'PASS' if checks[name] else 'FAIL'}] {name}")
        print(f"D8 final totals: {passed} passed, {failed} failed")

        return 0 if failed == 0 else 1

    finally:
        agent.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"\n[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
