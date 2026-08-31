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
| leaks detected | 9/12 | **11/12** |
| line localised | 7/12 | **8/12** |
| leak type correct | 7/12 | **8/12** |
| false positives | 0/8 | 0/8 |
| localisation precision | **63.6%** | 61.5% |
| candidates reported on injected cases | 11 | 13 |

The localisation sets reconcile exactly: pipeline gets `l01 l02 l04 l06 l07 l10
l11`; the agent gets those same seven **plus `l05`**, and loses none.

Per case, where the two differ:

| case | pipeline | agent | why |
|---|---|---|---|
| `l05_full_sample_zscore` | missed | **detected, localised, typed** | triage picks `expanding_stat`, which runs clean and moves Sharpe **+0.162** — the wrong way. The pipeline stops there. The agent retries and `rolling_stat` proves it at **−0.290**, on the ground-truth line. |
| `l04_htf_merge` | 3 candidates | 2 candidates | one fewer redundant candidate on the same leak; still localised and typed. |
| `l06_backward_fill` | 1 candidate | 2 candidates | the agent also proves the `L07` resample candidate at line 8; the correct type is still found at line 10. |
| `l08_forward_target` | missed | **detected**, not localised | triage's `drop_column` and `trailing_window` do not apply at line 7; the sweep reaches `future_shift`, which turns the forward-looking `forward_max` into a backward one and deflates Sharpe **2.575 → 0.433**. The repair is correct and the leak is real; the case file records line 9, so it scores as detected but not localised. |

**Spread over three passes: none.** detected `[11, 11, 11]`, localised
`[8, 8, 8]`, false positives `[0, 0, 0]`. The caches were deleted first and each
pass ran against its own, issuing 57 live prompts apiece, so all three paid for
their own answers rather than replaying the first. At temperature 0 the agent is
stable on this suite. That is a measurement, not an assumption; `--repeat`
rotates the cache directory precisely so it cannot become one.

**Cost.** 57 LLM calls per pass across 20 audits, against a ceiling of 50 calls
and 60 sandbox runs per audit.

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

## Session 6 — case 21 (freqtrade #11346) and the leak-signature memory

### The flagship case: a real bug the official tool passes

Case 21 is not ours. It is [freqtrade issue #11346](https://github.com/freqtrade/freqtrade/issues/11346),
reported by a freqtrade user whose strategy backtested beautifully and lost
money live, and whose own look-ahead analysis said `has_bias: No`.

It is added as `eval/cases/htf_merge_11346` with `"frozen": false`, which keeps
it out of every suite and every published denominator. The frozen 20 still read
20/12/8, and nothing above this entry moves.

The generic freqtrade shim could not be used for it. That shim resamples the
higher timeframe outside freqtrade and hands the engine one pre-merged frame,
so it never runs freqtrade's own informative path - which is where the bug
lives. `eval/baselines/freqtrade_11346.py` and the case's own `ft_strategy.py`
were built separately: a native `IStrategy` that declares `informative_pairs()`
and pulls the weekly frame through `self.dp.get_pair_dataframe()`.

**Head to head, same file, same window, same data:**

| | verdict | Sharpe |
|---|---|---|
| **Hindsight agent** | leak proven by execution | 3.589 → **0.461** |
| **freqtrade lookahead-analysis** | `has_bias: No`, `biased_indicators` empty | — |

The leaked native strategy wins **95.0% of 20 trades**; repaired, it wins 50.0%
of the same 20. freqtrade calls both of them clean.

**Why it misses, from freqtrade's installed source rather than from inference.**
`optimize/analysis/lookahead.py:146-148` truncates each re-run to the entry
candle plus exactly one base candle, and `data/dataprovider.py::historic_ohlcv`
loads the informative series against that same truncated range, filtering
candles by their **open** date. A weekly candle stamped Monday therefore
survives a truncation to Wednesday, and survives it *whole* - the stored candle
already contains Friday's close. Both runs read the identical value, so nothing
differs and nothing is reported. This is structural: no truncation on open-date
boundaries can expose a leak that lives inside a single higher-timeframe candle.

**A correction to our own framing.** This case was scoped as "reproduce #11346
using `@informative` or `merge_informative_pair`". The reporter used neither -
their strategy has no `informative_pairs` method at all. `merge_informative_pair`
is *safe*; it shifts the informative stamp forward by one interval, and that is
precisely why bypassing it leaks. So the leak could not honestly be put inside
that helper. `ft_strategy.py` uses the native informative path for everything
except the merge and then hand-rolls the merge as the reporter did;
`ft_clean.py` restores `merge_informative_pair()`, which is the fix.

**A third instance of the localisation caveat.** The agent repaired line 16
(`weekly = ...`) rather than line 20 (`w_up = ...`), which `meta.json` names.
The two are algebraically identical - shifting `weekly` propagates through
`weekly_sma` - and the agent landed on Sharpe 0.461, to three decimals the
reference `clean.py` value. It is scored **not localised** anyway, and the
frozen line was not changed to match.

**A session-2 question, settled.** `eval/baselines/freqtrade.py` documents an
artifact where raw `has_bias` fires `True` on nearly every case, clean ones
included, and concludes `biased_indicators` is the signal to trust. That
artifact is **shim-specific**: on the native strategies freqtrade reports
`has_bias = False` for both variants, with no timing noise at all. It came from
the shim turning a 0/1 position series into discrete enter/exit events.
`biased_indicators` was used as the cross-check regardless, and it is empty for
both. Both signals agree, and both are wrong.

### Memory: recognition, and deliberately nothing more

`hindsight_core/memory.py` stores confirmed leak signatures as a flat,
human-editable JSON list keyed by `leak_type` plus the normalised snippet. File
and line are excluded from the signature: the same bug pasted into another
module is the same bug.

A hit skips **one** thing - the triage LLM call - and supplies the operation
that worked last time. It does not skip the prover, the sandbox, or
`hooks/verification.py`. `record()` is reachable only from the branch that has
already passed the gate, so nothing enters the store unproven and nothing
leaves it as proof.

That claim is tested, and the test was checked by mutation rather than trusted:
an orchestrator that appends a `Finding` built straight from a stored entry
makes `test_a_memory_hit_still_has_to_be_proven_by_execution` fail on the
verification hook with *"cites before run ... which is not in the run store"*.

**Two things wiring it up exposed, both fixed:**

- The core test suite was order-dependent and quietly self-confirming. One
  test's audit taught the shared default store a signature that the next test's
  audit then recognised, skipping the triage that test existed to exercise.
  `tests/core/conftest.py` now gives every core test an empty store.
- The eval would have stopped reproducing. A warm store makes case N's result
  depend on cases 1..N-1 and on every audit ever run on the machine.
  `eval/detectors.py` now scores each case against a **cold** store. Memory's
  speed benefit is a product feature, not something the benchmark gets to bake
  in silently.

**What memory does not do.** It has not been measured as a speedup. The honest
statement is that it removes one LLM call per recognised candidate; no
end-to-end timing claim is made here, because the eval deliberately runs cold
and so never exercises it.

---

## The numbers that hurt

### The bug that cost a false positive and two localisations, and how it was found

The first agent run scored 11/12 detected but **1/8 false positives** and only
6/12 localised. One defect explained all three losses.

`c06_timeseries_split_pipeline` is a clean control: a legitimate ML pipeline
with a chronological `TimeSeriesSplit` and an embargoed last training row. Its
line 30 is

```python
target = (df["close"].shift(-1) > df["close"]).astype(int).reindex(features.index)
```

That is a supervised training label. It is forward-looking **by definition** —
that is what a label is — and it never reaches the decision except through
`.fit()` on training folds. The scanner flagged the `shift(-1)` (correctly: it
is tuned for recall). Triage called it a leak (incorrectly). Every operation
triage proposed failed to apply, and the mechanical sweep reached
`future_shift`, which flips the label. The fitted model collapsed, Sharpe fell
0.994 → 0.822, and differential execution reported a proven leak.

**A falling Sharpe is a weaker fact than it looks.** Corrupting a legitimate
input deflates a strategy exactly the way removing a leak does, and the
execution test cannot tell the two apart.

It also *hid* the real leaks. On `l11`, measured directly — one variable
changed, everything else identical:

| | Sharpe | delta against its own baseline |
|---|---|---|
| `l11` untouched | 2.0305 | — |
| ⤷ `fit_in_fold` at line 31 (the real leak, ground-truth line) | 1.1173 | **−0.913 → proven** |
| `l11` with the label flipped (`future_shift` at line 28) | 0.3530 | — |
| ⤷ `fit_in_fold` at line 31 | 0.3530 | **−0.0000 → no effect** |

Not merely weakened — annihilated to exactly zero. A `SelectKBest` fitted on an
uninformative label picks the same four features whether or not it saw the test
fold, so the leak it exists to have becomes unmeasurable. `l10` fails the same
way: after the label patch takes Sharpe 3.438 → 0.209, removing the actual
`shuffle=True` measures nothing.

### The fix: reachability, not a prompt

`scan_file` now follows where a forward-looking value goes. **A value absorbed
by `.fit()` is training data; a value that reaches the decision by any other
route is a feature.** Least fixed point outward from the fit calls, so the label
is still recognised when it passes through `train_test_split` first, as in
`l10` — hard-coding that call name would have made this a rule about
scikit-learn's API instead of a rule about where information flows.

Measured across all twenty cases, before and after the exemption, with
everything else held constant:

| | detected | localised | type | false positives | precision |
|---|---|---|---|---|---|
| agent, before | 10/12 | 7/12 | 7/12 | 1/8 | 58.3% |
| agent, after | 10/12 | **8/12** | **8/12** | **0/8** | **66.7%** |

`c06` returns to clean — both remaining candidates are triaged as not leaks and
nothing is proven. `l10` now proves its real leak at the ground-truth line 25
(`shuffle=True` → `shuffle=False`, Sharpe 3.438 → 0.974, delta **−2.463**), and
`l11` keeps line 31. The pipeline's numbers are unchanged by the exemption,
because it never proved those label candidates anyway — it gives up after one
attempt.

**The discrimination is what matters.** `l08`'s `forward_max` and `c03`'s
reporting column are built identically to a label and are *not* exempted,
because neither reaches a `.fit()` call. The rule cannot key off the shift.

### The removed experiment: LLM self-critique of its own findings

A repair the model never proposed — one the mechanical sweep found by trying
operations until the number moved — was made to answer for itself before it
could become a finding, with the diff and the delta in front of it: *did this
remove information the strategy was not entitled to, or corrupt something it
was?*

It was built, measured, kept for one revision, and then deleted. The full
ledger, all on the twenty frozen cases:

| | detected | localised | type | false positives | precision |
|---|---|---|---|---|---|
| before the gate, before the reachability rule | 11/12 | 6/12 | 6/12 | 1/8 | 46.2% |
| with the gate, before the reachability rule | 10/12 | 7/12 | 7/12 | 1/8 | 58.3% |
| with the gate, with the reachability rule | 10/12 | 8/12 | 8/12 | 0/8 | 66.7% |
| **gate deleted, reachability rule kept** | **11/12** | **8/12** | **8/12** | **0/8** | 61.5% |

**The gate gave opposite answers to the same question.** On `l11` it correctly
rejected the label patch ("modified the supervised training label, which is
inherently forward-looking"). On `c06` and `l10` it accepted the identical
construction. On `l08` it rejected a genuine repair — `forward_max` is not a
label, it feeds the signal directly — and cost a detection.

Once `scan_file` handled the label cases deterministically, the gate had nothing
left to catch. Across the three-pass run it fired **six times, all six on `l08`,
and every one was wrong**: its entire remaining effect on the suite was to
suppress one correct finding. Deleting it returned `l08` to detected with false
positives unchanged at 0/8 — the risk it guarded against did not materialise.

The one cost is localisation precision, 66.7% → 61.5%, and that is arithmetic
rather than a new wrong candidate: the numerator is unchanged at 8, the
denominator goes 12 → 13 because `l08`'s proven candidate sits at line 7 while
the case file records line 9.

**The lesson, stated plainly:** an LLM judgement inside the proof path is the
kind of confident, plausible, unverifiable claim this project exists to refuse.
It was wrong half the time on the one question it was asked, and the
deterministic rule that replaced it is right every time by construction.

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
- **`l08` is detected but not localised.** The agent proves `L01@7` — turning
  `rolling(10).max().shift(-10)` into `.shift(10)`, Sharpe 2.575 → 0.433 — which
  is a correct repair of a genuinely forward-looking column that feeds the
  signal. The case file records the leak at line 9 with type `L08`, so it scores
  detected, not localised, not typed. Line 9 itself remains the vocabulary
  boundary described above.
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

# Case 21 (freqtrade #11346) - outside the frozen 20, reached by name
hindsight eval --case htf_merge_11346 --detector agent
python -m eval.baselines.freqtrade_11346     # needs freqtrade installed
```

Case 21's combined record is `eval/results/case21.json`; the freqtrade half is
`eval/baselines/results/freqtrade_11346.json`. The freqtrade baseline is a
development dependency only - it is never imported by `hindsight_core` and is
not needed to run an audit.

`--repeat N` gives each pass its own prompt cache, so a repeat costs real
provider calls. Sharing one cache would replay the first pass byte for byte and
report a spread of zero however unstable the agent actually was.

Every audit writes a full trajectory to `runs/<audit_id>.json`: the verdict, the
budget spent, every finding with both run IDs, every unproven candidate with the
reason it stayed unproven, and the complete event stream.
