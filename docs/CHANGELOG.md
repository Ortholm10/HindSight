# Improvement Changelog

Every number here was produced by running `hindsight eval` against the twenty
frozen cases. Nothing is estimated and nothing is rounded in our favour.

---

## Session 5 — the agent loop

**What changed.** `hindsight_core/pipeline.py` walks the candidate list once, in
scan order, asks one question per candidate, and stops. It never re-runs the
baseline after a repair, never asks whether the number it just produced is
believable, and never tries a second operation when the first one fails.
`hindsight_core/orchestrator.py` does all three. Both call the same `triage()`
function, the same prover primitives and the same sandbox, so the difference
between the two columns below is the loop and nothing else.

### The table

`hindsight eval --detector pipeline` and `--detector agent --repeat 3`, twenty
frozen cases, three years of daily bars.

| | pipeline | agent |
|---|---|---|
| leaks detected | 9/12 | **10/12** |
| line localised | 7/12 | 7/12 |
| leak type correct | 7/12 | 7/12 |
| false positives | **0/8** | 1/8 |
| localisation precision | **63.6%** | 58.3% |
| candidates reported on injected cases | 11 | 12 |

Per case, where the two differ:

| case | pipeline | agent | why |
|---|---|---|---|
| `l05_full_sample_zscore` | missed | **detected, localised, typed** | triage picks `expanding_stat`, which runs clean and moves Sharpe **+0.162** — the wrong way. The pipeline stops there. The agent retries and `rolling_stat` proves it at **−0.290**, on the ground-truth line. |
| `l04_htf_merge` | 3 candidates | 2 candidates | one fewer redundant candidate on the same leak; still localised and typed. |
| `l06_backward_fill` | 1 candidate | 2 candidates | the agent also proves the `L07` resample candidate at line 8; the correct type is still found at line 10. |
| `c06_timeseries_split_pipeline` | clean | **false positive** | see below. |
| `l10_random_split` | localised at line 25 | detected, not localised | see below. |

**Spread over three passes: none.** detected `[10, 10, 10]`, localised
`[7, 7, 7]`, false positives `[1, 1, 1]`. Each pass ran against its own prompt
cache — 88, 79 and 78 prompts respectively — so all three genuinely paid for
their own answers rather than replaying the first. At temperature 0 the agent is
stable on this suite. That is a measurement, not an assumption; `--repeat`
rotates the cache directory precisely so it cannot become one.

### The behaviours that earn the loop

**Multi-step repair.** `l05` is the guaranteed instance and it was not staged:
`expanding_stat` is listed first in `.claude/skills/l05.md`, is taxonomically
valid, and does not work. Seeded instead with `lag` — also a defensible triage
answer — the chain is three steps driven entirely by execution: `lag` crashes
(`AttributeError: 'numpy.float64' object has no attribute 'shift'`, because
`quantile()` returns a scalar), `expanding_stat` runs clean at +0.162, and
`rolling_stat` proves at −0.290.

**Continued suspicion.** On `tests/fixtures/stacked_leaks.py` the baseline
descends 1.886 → 0.633 → 0.395 across two kept repairs. The second leak — a
full-sample `quantile` threshold — *raises* Sharpe by 0.62 when removed from the
untouched file, so the single-pass pipeline files it as no-effect and moves on.
Measured against the repaired baseline it deflates by 0.237. The general rule
the loop encodes: **"no effect" is a claim about a baseline, not about a line of
code**, so every such verdict is withdrawn when the baseline it was measured
against is replaced.

**Inconclusive outcomes stay distinct.** "Leak proven", "my patch was broken"
and "zero trades after patching" are separate statuses with separate reporting,
and an audit that ends holding an unsettled candidate returns `inconclusive`,
never `clean`. Hitting an LLM-call or sandbox-run ceiling returns
`stopped_on_budget`, which is also never `clean`.

**The vocabulary boundary.** `l08` line 9 is
`signal = forward_max > close * 1.05`. The decision *is* the leaking
comparison: no operand can be dropped, and rebuilding it needs a substitute
column, which is a judgement rather than a removal. The agent reports "detected,
not mechanically patchable" instead of inventing a value to force a patch
through.

---

## The numbers that hurt

### One false positive, and it is the same root cause as two localisation losses

`c06_timeseries_split_pipeline` is a clean control: a legitimate ML pipeline
with a chronological `TimeSeriesSplit` and an embargoed last training row. Its
line 30 is

```python
target = (df["close"].shift(-1) > df["close"]).astype(int).reindex(features.index)
```

That is a supervised training label. It is forward-looking **by definition** —
that is what a label is — and it never reaches the decision except through
`.fit()` on training folds. The scanner flags the `shift(-1)` (correctly: it is
tuned for recall). Triage calls it a leak (incorrectly). Every operation triage
proposes fails to apply, and the agent's mechanical sweep eventually reaches
`future_shift`, which flips the label. The fitted model collapses, Sharpe falls
0.994 → 0.822, and differential execution reports a proven leak.

**A falling Sharpe is a weaker fact than it looks.** Corrupting a legitimate
input deflates a strategy exactly the way removing a leak does, and the
execution test cannot tell the two apart.

The same mechanism costs two localisations:

- `l10_random_split`: the label patch takes Sharpe 3.438 → 0.209. The real leak
  (`shuffle=True`, line 25) is then invisible — a model fed a broken label no
  longer responds to anything, so removing the real leak measures `no_effect`.
  Detected, but at the wrong line.
- `l11_preprocess_before_split`: the same shape, 2.031 → 0.353, before the gate
  below caught it.

The pipeline scores 0/8 here **because it gives up after one attempt**. Its
clean sheet on false positives is bought with the same fragility that loses it
`l05`. That is the trade the agent makes, stated plainly.

### The experiment: LLM self-critique of its own findings

A repair the model never proposed — one the mechanical sweep found by trying
operations until the number moved — was made to answer for itself before it
could become a finding, with the diff and the delta in front of it: *did this
remove information the strategy was not entitled to, or corrupt something it
was?*

Measured on all twenty cases, before and after:

| | detected | localised | type | false positives | precision |
|---|---|---|---|---|---|
| agent, no confirmation gate | 11/12 | 6/12 | 6/12 | 1/8 | 46.2% |
| agent, with the gate | 10/12 | 7/12 | 7/12 | 1/8 | 58.3% |

**The gate gives opposite answers to the same question.** On `l11` it correctly
rejected the label patch ("modified the supervised training label, which is
inherently forward-looking") and went on to prove the real `L11` leak at the
ground-truth line. On `c06` and `l10` it accepted the identical construction. On
`l08` it rejected a genuine repair — `forward_max` is not a label, it feeds the
signal directly — and cost a detection.

One correct call, one wrong call and two misses, across four instances of one
question. The gate is kept because it improves localisation and precision at no
cost in false positives, but it is **not** load-bearing and must not be
described as a safeguard. An LLM judgement inside the proof path is exactly the
kind of confident, plausible, unverifiable claim this project exists to refuse.

**The deterministic fix, not yet built:** a forward-looking name whose only
consumer is the target argument of `.fit()` is a training label and must not be
patched. That is static reachability, it is decidable, and it would address
`c06`, `l10` and `l11` without touching `l08`. It belongs in `scan_file`, not in
a prompt.

### What remains unproven

- **Zero slippage and zero commission.** Every number here assumes fills at the
  close with no cost. Real execution is worse. This is not leakage and the tool
  does not claim to measure it.
- **The universe is survivorship-biased.** The eight ETFs in the cache exist
  today. `l12_hindsight_universe` is a case about exactly this bias, and the
  data it runs on has it.
- **`l12` cannot be detected at all.** There is no L12 rule in `scan_file`, so
  the scanner emits no candidate on its ground-truth line. It is a miss with a
  known cause, not a near-miss.
- **`l03` and `l09` are detected but not localised**, and this is partly an
  argument with the ground truth rather than a failure. On `l03` the agent
  patches line 8 (`signal = close > sma`); `meta.json` names line 9, the
  unlagged `return`. Both repairs are correct and both remove the leak. On `l09`
  the agent names line 8 — `rolling(2*k+1, center=True)` inside the helper —
  while `meta.json` names line 24, the call site. The recorded lines were not
  changed to match: cases edited after seeing results are not evidence.
- **One seed, one window.** The sandbox fixes the RNG seed so a before/after
  delta is attributable to the patch rather than to the draw. A strategy whose
  result depends on the draw is therefore measured at one draw. The three-window
  robustness check lives in the case validity rules, not in the audit path.
- **Three passes is a small sample.** The spread is zero across three runs at
  temperature 0. It bounds run-to-run variance on this suite and this provider;
  it says nothing about a different model or a warmer temperature.

---

## Reproducing these numbers

```bash
hindsight eval --detector pipeline --json --out eval/results/pipeline.json
hindsight eval --detector agent --repeat 3 --json --out eval/results/agent.json
hindsight audit tests/fixtures/stacked_leaks.py --data eval/data/SPY.csv --mode agent
```

`--repeat N` gives each pass its own prompt cache, so a repeat costs real
provider calls. Sharing one cache would replay the first pass byte for byte and
report a spread of zero however unstable the agent actually was.

Every audit writes a full trajectory to `runs/<audit_id>.json`: the verdict, the
budget spent, every finding with both run IDs, every unproven candidate with the
reason it stayed unproven, and the complete event stream.
