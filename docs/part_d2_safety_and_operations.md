# Part D2 — Refund Agent Safety and Operations

This guide describes the current offline Part D2 implementation. It separates requirements in the supplied policy documents from behavior the code currently performs. The agent is an educational demonstration, not an operational refund service.

## Architecture and tool flow

The public compatibility module `part_d_refund_agent.py` re-exports the corrected implementation in `part_d_refund_agent_corrected.py`. `RefundAgent` reads order information from the project SQLite database, loads policy documents for retrieval, and uses the local `mock_llm.py` interface to select bounded tools. The refund-related tools include order lookup, policy search, refund calculation, a fail-closed refund decision, and escalation. Agent run logs and checkpoints are separate from the historical refund ledger.

Conceptually, the flow is:

```text
customer scenario
      |
bounded mock-model/tool loop
      |-- order lookup and policy retrieval
      |-- refund calculation from trusted order fields
      |-- issue_refund or escalation decision
      |
checkpoint + run log
```

The current `issue_refund` implementation is side-effect free with respect to financial systems and the refund ledger: it returns a fail-closed decision and does not record a new refund or move money. A normal agent run can still create or update a checkpoint under `state/` and append run events to `outputs/agent_runs.jsonl`; preserve those files as project evidence. This project has no payment integration or operational human-approval workflow.

### Historical refund evidence and demonstration wording

`outputs/agent_summary.json` is historical evidence from simulated behavior and reports two simulated refund ledger entries. It does not describe the current fail-closed refund-decision implementation and must not be interpreted as evidence that the current project approves, records, or pays refunds. Preserve that JSON report unchanged.

The local mock may produce demonstration wording that a refund was “sent to a human lead.” Such wording is simulated output only; it does not mean a human was contacted, that an approval queue exists, or that an approval occurred. The current implementation does not operationally route requests to a human.

## Policy requirements and implemented behavior

The supplied [`refund_policy.txt`](../data/policies/refund_policy.txt) states that:

- Only completed orders qualify; cancelled orders were not charged.
- A completed order delivered in more than 45 minutes qualifies for a full order-amount refund; 45 minutes or less does not qualify.
- Refunds up to ₹1,000 may be approved automatically under the policy. Refunds above ₹1,000 require a human support lead’s approval before money is returned.
- The customer must request a refund within seven days of delivery.
- The policy describes a credit to the original payment method within five working days after an approved refund.

The current implementation enforces completed-order status, requires delivery duration to be strictly greater than 45 minutes, calculates the full order amount, and compares valid monetary values using `Decimal` at cent precision. Invalid, non-finite, non-positive, or sub-cent amounts are rejected. The ₹1,000.00 boundary is distinct from ₹1,000.01: the latter is above the policy’s human-approval threshold.

### Seven-day limitation: fail closed

The available order data does not establish a verified delivery timestamp. `order_ts` is not treated as delivery time, and the implementation does not add `delivery_minutes` to it to create a delivery date. Therefore the seven-day request deadline cannot be checked. The agent flags `refund_deadline_unverifiable_from_order_data` and returns an escalation decision for an otherwise qualifying request; it does not automatically approve the refund or operationally route the request to a human. **This project does not claim seven-day policy compliance.**

The policy’s automatic-approval and human-approval rules describe the intended business process. The current code does not actually approve, record, pay, or operationally route a human approval; it fails closed and returns an escalation decision because the deadline is unverifiable.

## Ledger validation and duplicate handling

The configured default ledger path points to the existing project refund ledger for normal execution. The agent reads historical entries to detect prior simulated-refund records; it does not append new refund entries. Tests use temporary ledger paths.

- A matching prior simulated refund is classified as already recorded and escalated for review.
- A different recorded amount is classified as a conflict and escalated for review.
- Multiple entries are examined; a mixed or conflicting set is treated as a conflict.
- Malformed ledger lines are reported as integrity errors. The agent fails closed, and the original malformed bytes are not rewritten.
- Invalid requested or order amounts are rejected rather than rounded into a valid amount.

These safeguards do not turn the historical ledger into a payment record. All financial behavior remains simulated, and no new entry is recorded by the current `issue_refund` tool.

## D7 — Bounded fallbacks

The agent reports termination reason separately from result classification. A fallback result uses `actual_outcome == FALLBACK`; the `stop_reason` identifies why execution stopped:

- `FALLBACK_MAX_ITERATIONS` — maximum agent steps reached.
- `FALLBACK_TOKEN_LIMIT` — token budget reached.
- `FALLBACK_TIMEOUT` — time budget reached, including a zero-second budget before the first model call.
- `FALLBACK_MODEL_ERROR` — model call failed.
- `COMPLETED` — normal completion.

Fallback does not mean a refund succeeded. The D7 regression harness checks each fallback category and verifies no simulated refund entries were written.

## D8 — Checkpoint and resume

Each new run creates a checkpoint identified by run ID. Creating a checkpoint with an ID that already exists fails before overwriting that checkpoint. Resume reads and validates the selected checkpoint, updates that checkpoint, and reuses a persisted terminal tool result rather than asking the model to issue the terminal tool again. Corrupt checkpoint data is rejected without proceeding to a model call.

The D8 test interrupts a run, resumes it from temporary isolated state, and verifies preserved order/request context, a single terminal tool trace entry, persisted terminal-result reuse, no refund ledger entry, and `real_money_moved == False`.

The saved `outputs/d8_resume_summary.json` reports seven passed checks, while the Phase 2.3 verification record described below reports 15 checks passed. These counts may refer to different runs or check groupings; their relationship is not independently established. Both records are historical evidence and should be preserved without treating either count as a universal total.

### Crash-window limitation

There is a crash window after a tool executes but before its result is durably saved in the checkpoint. A restart during that interval may not know that the tool already ran. Reusing a terminal result after it has been persisted protects the tested interruption/resume path, but **exactly-once execution across arbitrary crashes is not guaranteed**.

## Test commands and verified Phase 2.3 results

Run from the project root:

```powershell
python -B test_part_d2_refund_rules.py
python -B test_part_d7_fallbacks.py
python -B test_part_d8_resume.py
```

`-B` disables Python bytecode writes. These harnesses isolate generated ledgers, run logs, checkpoints, and summaries in temporary directories; they read project inputs such as the SQLite database and policy files. The verified Phase 2.3 results were:

- D2 refund rules: **13 passed**.
- D7 fallback regression: **6 passed**.
- D8 checkpoint/resume: **15 checks passed**.

These are historical reported results from that verification record. They are not a claim that these commands were executed while preparing this documentation, nor a guarantee about later changes. The separate saved D8 summary records seven passed checks; see the note above.

## Current limitations and future enhancements

Current limitations:

- No verified delivery timestamp is available, so the seven-day deadline is unverifiable and approval remains fail-closed.
- No real refunds are recorded or transferred; there is no payment integration or operational human-approval workflow.
- The checkpoint mechanism does not provide exactly-once guarantees across arbitrary crashes.
- The agent relies on local project data, policy documents, and a mock model interface; it is not a production service.

Possible future work, subject to separate design and safety review:

- Add an authoritative, verified delivery-event timestamp and define its provenance.
- Design durable idempotency around tool execution and checkpoint persistence, then test crash recovery explicitly.
- Add a reviewed operational approval workflow and payment integration only with appropriate authorization and controls.
- Publish a dependency manifest and reproducible environment instructions.
