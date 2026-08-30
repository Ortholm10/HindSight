# Agent trajectories

One JSON file per audit, written by `hindsight_core/orchestrator.py` on every
run — including the runs that proved nothing. A record that keeps only the
successes is a highlight reel, and the audits where the agent was wrong are the
ones worth reading.

Each file holds:

| field | what it is |
|---|---|
| `verdict` | `leaks_proven`, `clean`, `inconclusive`, `stopped_on_budget`, or `untestable`. Never `clean` when a candidate was left unsettled or a ceiling was hit. |
| `reason` | one sentence saying why the loop stopped |
| `original_baseline` | the unmodified strategy's execution record |
| `budget` | LLM calls and sandbox runs spent, against their ceilings |
| `findings` | proven leaks, each citing `before_run_id` and `after_run_id` |
| `unproven` | every candidate examined and not proven, with its status, its reason, and every repair attempted |
| `events` | the full stream: `scan_complete`, `triage`, `baseline`, `prove_start`, `prove_result`, `agent_decision`, `final` |

The run IDs resolve against `.hindsight/runs/<run_id>.json`, which is where the
execution records themselves live. A finding whose run IDs are not in that store
is rejected by `hindsight_core/hooks/verification.py` before it can reach a
report.

`_before_confirmation_gate/` holds the four trajectories the changelog cites for
the removed-experiment section — `c06`, `l08`, `l10`, `l11` as they ran before
the sweep-confirmation gate existed. The full before/after numbers for all
twenty cases are in `eval/results/agent_before_gate.json`.
