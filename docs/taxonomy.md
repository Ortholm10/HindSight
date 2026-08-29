# Hindsight — Leak Taxonomy

**This file is the foundation of the project.** Three consumers read it:

1. `.claude/skills/` — one markdown file per type, read at runtime by `hindsight_core` as the agent's leak knowledge.
2. `eval/inject.py` — the injection recipes below are implemented literally.
3. `eval/cases/` — the ground truth. Detection is scored against the type IDs and line numbers defined here.

Nothing in the build starts before this file is settled.

---

## 1. The invariant

> **At every row `t`, a feature may only depend on data available at or before the decision point of row `t`.**

Every leak in this document is a violation of that one sentence. Every fix restores it by *removing* access to information, never by adding a value.

Two corollaries that decide the edge cases:

- **The decision point matters, not the bar.** A close-derived indicator is legitimate for a decision executed after that close, and illegitimate for a decision executed at that bar's open. The same line of code is a leak or not depending on when the order fills. This is the core reason detection alone is insufficient and execution proof is required.
- **A leak is only a leak if it reaches the signal path.** A forward-looking column computed for reporting, plotting, or post-hoc analysis and never referenced by the trading decision is not a leak. Several clean controls exploit exactly this, and a scanner without dataflow awareness will flag all of them.

---

## 2. The strategy contract

Every eval case exposes one function:

```python
def run_strategy(df: pd.DataFrame) -> pd.Series:
    """df has columns open/high/low/close/volume, DatetimeIndex, sorted ascending.
    Returns an equity curve indexed identically to df."""
```

Most cases implement this as a vectorised pandas backtest. A minority wrap `backtesting.py` internally. The sandbox and prover only know the contract.

**Why not `backtesting.py` for everything:** its default fill is the *next* bar's open, which silently corrects the canonical same-bar alignment bug (L03). Cases must be able to express the bug, so the vectorised form is primary.

**Reference vectorised form**, correct:

```python
signal   = (df['close'] > df['close'].rolling(20).mean())
position = signal.shift(1).fillna(False).astype(int)   # decided on t-1, held over t
returns  = position * df['close'].pct_change()
equity   = (1 + returns).cumprod()
```

Deleting the `.shift(1)` on line 2 is L03. That single line is the project's canonical example.

**Metrics:** annualised Sharpe (primary), total return, max drawdown. Sharpe on zero-variance or zero-trade returns is undefined — return the `ZERO_TRADES` outcome, never `0.0` or `NaN` silently.

---

## 3. Summary table

| ID | Name | Signature | Fix class | Severity | Detect difficulty | Freqtrade-expressible |
|---|---|---|---|---|---|---|
| L01 | Explicit future index | `.shift(-n)`, `iloc[i+1]` | lag | Extreme | Trivial | Yes |
| L02 | Centered rolling window | `rolling(center=True)` | trailing window | High | Easy | Yes |
| L03 | Same-bar execution mismatch | missing `.shift(1)` on signal | lag | High | Medium | Partly |
| L04 | HTF merge without closure lag | `resample` / `merge_asof` | lag HTF series | Medium | **Hard** | Yes |
| L05 | Full-sample normalisation | `.mean()`/`.std()` on whole column | expanding/rolling | Medium | Medium | Partly |
| L06 | Backward fill | `.bfill()` | ffill or drop | Medium | Easy | Yes |
| L07 | Resample label/closed misalignment | bare `.resample()` | explicit label + lag | High | **Hard** | Yes |
| L08 | Forward-window target as feature | `rolling().max().shift(-n)` | remove from signal | Extreme | Medium | No |
| L09 | Two-sided turning-point detection | `argrelextrema`, `find_peaks` | causal confirmation | Extreme | Easy | No |
| L10 | Non-chronological split | `train_test_split` without `shuffle=False` | chronological split | High | Easy | No |
| L11 | Preprocessing fitted before split | `.fit(X_all)` then split | fit inside fold | Medium | Medium | No |
| L12 | Hindsight universe selection | filter symbols by full-period stat | point-in-time universe | Medium | Hard | No |

**Mapping to published work.** L01–L09 are temporal leakage, corresponding to Kapoor & Narayanan's L3.1. L10 corresponds to their L1.1 (no clean train/test separation). L11 corresponds to L1.2 and L1.3 (preprocessing and feature selection on the full dataset). L08 corresponds to L2 (illegitimate features). L12 corresponds to L3.3 (sampling bias in the test distribution). Citing that mapping in the README is what makes the case set defensible against "you invented your own bugs."

**The 12 injected cases are one per type.** Two types get a harder second instance (L04 and L07) added in Session 9, outside the frozen 20.

---

## 4. Type cards

Each card is the source for one `.claude/skills/` file and one injection recipe.

---

### L01 — Explicit future index

**What it is.** The code names a future row directly.

**Why it inflates.** The signal is the answer. Sharpe typically exceeds 5 and the equity curve is close to monotonic — the tell is that it looks *too* good rather than slightly good.

**Detection pattern.**
- `.shift(` with a negative numeric literal
- `.iloc[i + k]` / `[i+1]` with positive offset inside a signal function
- `.diff(-n)`, `.pct_change(-n)`

**Fix.** Negate the shift, or remove the feature from the signal path.

**Injection recipe.** From the reference form, replace the signal with `df['close'].shift(-1) > df['close']`. Ground truth = that line.

**False-positive trap.** `.shift(-1)` used to build a forward-return column for reporting only. See C03. **The scanner must not treat a negative shift as conclusive** — dataflow to the signal is what makes it a leak.

**Why it is in the set.** It calibrates the low end. Every system should catch it. A baseline that misses L01 is broken, not merely weak.

---

### L02 — Centered rolling window

**What it is.** `rolling(n, center=True)` places the label at the middle of the window, so roughly `n/2` future bars enter every value.

**Why it inflates.** Produces a smoothed series that anticipates turns. Inflation scales with window size.

**Detection pattern.** `rolling(` with `center=True`. Also `scipy.signal.filtfilt` (zero-phase, forward-backward) and `savgol_filter` applied to the full series.

**Fix.** Remove `center=True` — trailing is the pandas default.

**Injection recipe.** Flip the flag on an existing SMA in the reference form.

**False-positive trap.** A centered window used to construct a chart overlay, never in the signal.

**Note.** One-shot LLM baselines frequently miss this, because the line reads as idiomatic smoothing.

---

### L03 — Same-bar execution mismatch

**What it is.** A signal derived from bar `t`'s close (or high/low) drives a position held *over* bar `t`. The canonical case: a missing `.shift(1)` between signal and position.

**Why it inflates.** The strategy acts on information from the end of the bar it is trading through. Moderate but very realistic inflation — this is the 4.12 → 0.34 example.

**Detection pattern.** A boolean or numeric column derived from `close`, `high`, `low`, or any indicator over them, assigned to a position and multiplied by a *same-index* return. The AST signature is the *absence* of a shift between derivation and use, so this rule is inherently over-productive and must not be tuned for precision.

**Fix.** `.shift(1)` on the signal before it becomes a position.

**Injection recipe.** Delete the `.shift(1)` from a clean vectorised case.

**False-positive trap.** C01 — the correct version is textually one token different. This pair is the most important discriminator in the eval, because it is where a static scanner and an execution prover give different answers.

**Engine note.** Not directly expressible in `backtesting.py` without contortion, since it fills at the next open by default. Mark partly-expressible for the freqtrade baseline and state the reason in the results table rather than scoring it as a miss.

---

### L04 — Higher-timeframe merge without closure lag

**What it is.** A daily or intraday strategy pulls in a weekly or 4-hour indicator, merged on a timestamp that falls *inside* the higher-timeframe bar. The HTF value used at row `t` includes bars that had not closed at `t`.

**Why it inflates.** Modest and plausible — often lifting Sharpe from around 0.8 to around 1.6 rather than to absurdity. That plausibility is precisely why it survives review.

**Detection pattern.**
- `.resample(...)` producing an HTF series later joined back to the base frame
- `pd.merge_asof(...)` where the right frame is coarser
- any join on a truncated timestamp key (`floor('D')`, `to_period('W')`)

without a shift applied to the HTF series before the merge.

**Fix.** Shift the HTF series by one full HTF period before merging, so row `t` sees only the last *closed* higher-timeframe bar. Equivalently, key the merge on the HTF bar's close time rather than its open time.

**Injection recipe.** Build a daily strategy filtering on a weekly SMA; merge the weekly series onto daily rows without lagging it.

**False-positive trap.** C04 — `merge_asof(direction='backward')` with a correct prior shift. `direction='backward'` alone does **not** make the merge safe: it finds the most recent HTF row at or before `t`, and that row is the currently-open bar whose value already reflects data through `t`. A detector treating `direction='backward'` as sufficient will pass the leaked version.

**Why it is the flagship.** Freqtrade's `lookahead-analysis` has documented false negatives on exactly this pattern (issues #11346 and #12507), and its own documentation concedes a negative result is not a guarantee. Session 9 reproduces #11346 as case 21.

---

### L05 — Full-sample normalisation

**What it is.** A feature is scaled by a statistic computed over the entire series — mean, standard deviation, min, max, quantile.

**Why it inflates.** The scaled feature encodes where each point sits relative to the *whole history including the future*. A z-score of +2 in 2019 already knows the 2020–2024 distribution.

**Detection pattern.**
- `(x - x.mean()) / x.std()` on a full column
- `.min()`, `.max()`, `.quantile()` on a full column feeding a threshold
- `StandardScaler().fit_transform(X)` / `MinMaxScaler()` on the full frame

**Fix.** `expanding().mean()` and `expanding().std()` (past-only, requires a warm-up period), or a trailing `rolling(n)` window. Both are causal.

**Injection recipe.** Replace a rolling z-score with a full-sample z-score.

**False-positive trap.** C02 uses `expanding()` and C07 uses `.max()` for a plot axis. `expanding()` is causal by construction and must never be flagged.

**Severity note.** Inflation here is often *modest*. That makes L05 the best test of the agent's continued-suspicion behaviour: fixing a loud leak first, then judging that the remaining number is still implausible.

---

### L06 — Backward fill

**What it is.** `.bfill()` fills a gap with the *next* known value, moving information backwards in time.

**Why it inflates.** Every filled row carries a future observation. Impact scales with gap frequency — sparse fundamental or macro series are the usual victims.

**Detection pattern.** `.bfill()`, `fillna(method='bfill')`, `fillna(method='backfill')`, `reindex(..., method='bfill')`, and `interpolate()` with a two-sided method on a signal-path column.

**Fix.** `.ffill()`, or drop the rows.

**False-positive trap.** C05 uses `ffill`. A genuinely benign `bfill` on static metadata (sector, currency) is worth adding as a variant if time allows.

---

### L07 — Resample label/closed misalignment

**What it is.** `df.resample(freq).agg(...)` stamps each aggregated bin with a boundary label. When the label is the bin's *left* edge, the aggregate of the whole period appears at the moment that period began.

**Why it inflates.** Every row of the resampled series is the completed statistic of a period that had not yet happened. Effect is strong and the code looks entirely ordinary.

**Detection pattern.** `.resample(` called without explicit `label=` and `closed=` arguments, where the result feeds a signal. **Pandas defaults differ by frequency** — several calendar frequencies default to right-labelled while others default to left — and that inconsistency is itself the trap. Flag any bare `resample` on the signal path rather than trying to encode the per-frequency defaults.

**Fix.** Set `label='right', closed='right'` **and** shift by one period, so row `t` sees the last fully closed bin. Setting the label alone is insufficient: a right-labelled bin is stamped at the instant it closes, which is still simultaneous with the decision.

**Injection recipe.** Compute a weekly mean via bare `resample('W')`, forward-fill onto daily rows, use as a filter.

**False-positive trap.** C08 — explicit `label='right', closed='right'` plus a shift.

**Note.** Closely related to L04, and the two co-occur in real code. Keep them distinct types, because the fixes differ.

---

### L08 — Forward-window target used as feature

**What it is.** A column built from a forward-looking window — forward max, forward return, triple-barrier label — is used in the trading decision rather than only as a training target.

**Why it inflates.** Extreme. The feature is a transformed copy of the answer.

**Detection pattern.** `rolling(n).max().shift(-n)`, `[::-1]` reversal followed by a rolling operation, and any function named `label`, `target`, `barrier`, or `future_` whose output reaches the signal.

**Fix.** Remove it from the signal path. Keep it for evaluation if it is a legitimate training target under a chronological split.

**False-positive trap.** C03 — the same construction, used only in a reporting column. This is the hardest control in the set, and the one most likely to be flagged by every baseline.

---

### L09 — Two-sided turning-point detection

**What it is.** Peak or trough detection that identifies an extremum by comparing a point to its neighbours *on both sides*.

**Why it inflates.** A peak cannot be known until after it has passed. Marking it at its own timestamp is time travel. Effect is extreme.

**Detection pattern.** `scipy.signal.argrelextrema`, `find_peaks`, `zigzag`, and any hand-rolled comparison of `x[i]` against both `x[i-k]` and `x[i+k]`.

**Fix.** Replace with causal confirmation — a peak is declared only after `n` bars have failed to exceed it — accepting the lag that entails. If no causal equivalent exists, remove.

**Note.** Easy to detect once the function name is known, but one-shot LLM baselines often accept it as legitimate technical analysis, which makes it a useful separator.

---

### L10 — Non-chronological train/test split

**What it is.** A supervised model over time-series features, split randomly rather than chronologically.

**Why it inflates.** Test rows are interleaved with training rows minutes apart and are near-duplicates. Reported accuracy is wildly optimistic, and the strategy built on it inherits the optimism.

**Detection pattern.** `train_test_split(` without `shuffle=False`; `KFold`, `cross_val_score`, or `GridSearchCV` without `TimeSeriesSplit` on a time-indexed frame.

**Fix.** Chronological split, or `TimeSeriesSplit`.

**False-positive trap.** C06 uses `TimeSeriesSplit` correctly.

**Scope note.** This is the type that most clearly supports the README's "any time-series pipeline" framing — identical in demand forecasting and predictive maintenance.

---

### L11 — Preprocessing or feature selection fitted before the split

**What it is.** A scaler, imputer, PCA, or feature selector is fitted on the full dataset, then the data is split.

**Why it inflates.** Test-set statistics are baked into the training representation. Milder than L10, but very common and rarely noticed.

**Detection pattern.** `.fit(` or `.fit_transform(` on a frame subsequently passed to a split function; `SelectKBest`, `RFE`, or correlation-based feature selection computed on the full frame.

**Fix.** Fit inside the training fold only — in practice, wrap in a `Pipeline` and let the CV splitter handle it.

**False-positive trap.** C06 — scaler inside a `Pipeline` inside `TimeSeriesSplit`.

---

### L12 — Hindsight universe selection

**What it is.** The instrument set is chosen using information from the end of the period — surviving tickers, current index constituents, or symbols filtered by full-period return or liquidity.

**Why it inflates.** The strategy is only ever tested on instruments that did well enough to still exist.

**Detection pattern.** A symbol list filtered by an aggregate over the full period; a hardcoded constituent list applied to a historical range.

**Fix.** A point-in-time universe, which usually requires data the user does not have.

**Honest limitation — state this in the report.** L12 is frequently **detectable but not patchable**. Hindsight reports it as an unproven-but-suspected finding, clearly separated from execution-proven findings, and never counts it in the proven column. This is the one type where the verification hook cannot be satisfied, and pretending otherwise would violate the project's own rule. It belongs in the "what remains unproven" section rather than the findings list.

---

## 5. The eight clean controls

Each control is a deliberate near-miss of a specific type. Their purpose is to make a naive scanner look bad and a prover look good. **These matter as much as the injected cases.**

| ID | Traps | What it does | Why it is clean |
|---|---|---|---|
| C01 | L03 | `rolling(20).mean()` signal with correct `.shift(1)` | One token from the leaked version; the shift is present |
| C02 | L05 | `expanding().mean()` / `expanding().std()` z-score | Expanding windows are past-only by construction |
| C03 | L01, L08 | `.shift(-1)` forward return, used only in a reporting column | Never reaches the signal path — requires dataflow analysis |
| C04 | L04 | `merge_asof(direction='backward')` with HTF series pre-shifted | The lag is applied before the merge |
| C05 | L06 | `.ffill()` on price gaps | Forward fill is causal |
| C06 | L10, L11 | `TimeSeriesSplit` with scaler inside a `Pipeline` | Chronological, and fitting is per-fold |
| C07 | L05 | `df['close'].max()` used for a chart axis limit | Not a feature; never enters the decision |
| C08 | L07 | `resample('W', label='right', closed='right')` plus a shift | Sees only closed bins |

**C03 and C07 are the two dataflow controls**, and they trap different signatures — a negative shift and a full-sample aggregate. Both are clean for the same underlying reason: the suspicious column never reaches the decision. Between them the dataflow argument is covered twice over, which is why the set stays at eight rather than adding a third.

**C03 is the hardest and the most valuable.** A scanner keyed on `.shift(-1)` flags it. A prover shifts the reporting column, re-runs, observes *no change in the equity curve*, and correctly discards the candidate. That single case is the clearest demonstration of why execution proof beats pattern matching — call it out by name in the video.

---

## 6. Case validity rules

Session 1 must assert these for every injected case, as tests. A case failing any of them is not evidence and must be repaired or replaced **before the set is frozen**.

### Why a raw Sharpe threshold is not enough

On one year of daily bars, the standard error of an annualised Sharpe ratio is approximately **1.0** — larger than most of the effects being measured. Three years brings it to roughly 0.58, five years to 0.45.

That figure does *not* invalidate the paired comparison. Clean and leaked versions run on identical data, so the difference between them is deterministic, not a sample statistic. But it does mean a small delta observed on one window is not evidence the leak matters — it may not survive a different window. The validity rules below test for robustness rather than for a single number clearing a bar.

### The rules

1. **Both versions run.** Clean and leaked each produce a non-empty equity curve with at least 10 position changes.

2. **Trade count is stable.** Position changes in the leaked version are within ±25% of the clean version. If the injection drastically changes how often the strategy trades, the two versions are different strategies and the Sharpe comparison is meaningless.

3. **The leak inflates, materially.** Both must hold:
   - **Relative:** leaked Sharpe ≥ 1.4 × clean Sharpe (or, where clean Sharpe ≤ 0.2, leaked ≥ clean + 0.5 in absolute terms — ratios are unstable near zero)
   - **Absolute:** leaked Sharpe − clean Sharpe ≥ 0.4

4. **The effect is robust across windows.** Split the data into three non-overlapping periods. The sign of the inflation must hold in **all three**, and the magnitude must clear rule 3 in at least two. This is the rule that does the real work — it is what separates a genuine leak from a lucky window, and it is the answer to a judge asking whether a result is noise.

5. **The fix restores.** Applying the documented fix to the leaked version reproduces the clean version's Sharpe within 0.05, on every window. This proves the fix is the true inverse of the injection, not merely something that lowers the number.

6. **Ground truth is exact.** The recorded file and line number point at the injected line, and the recorded type ID matches.

For the eight controls, assert instead: applying *any* candidate fix leaves the equity curve unchanged within tolerance, on every window. That is the property the prover relies on.

### Data window

**Three years of daily bars, not one.** This is a change from the earlier plan and it is close to free: 750 rows instead of 250 is negligible compute, the cache is still small enough to commit, and it buys two things that matter — Sharpe standard error drops from about 1.0 to about 0.58, and there is enough data to cut the three non-overlapping windows rule 4 requires. Fix the date range explicitly in the case metadata so it never drifts.

## 7. Fix vocabulary

Every patch the agent may apply. The list is closed — a proposed patch outside it should be rejected, because CLAUDE.md rule 2 forbids introducing a value the data did not contain.

| Operation | Applies to |
|---|---|
| Lag a series by `n` periods | L01, L03, L04, L07 |
| Replace two-sided window with trailing | L02, L09 |
| Replace full-sample statistic with `expanding()` or `rolling()` | L05 |
| Replace backward fill with forward fill or drop | L06 |
| Add explicit `label`/`closed` plus lag | L07 |
| Remove a column from the signal path | L01, L08, L09 |
| Move `.fit()` inside the training fold | L11 |
| Replace random split with chronological | L10 |
| **Report only, no patch** | L12 |

Every operation removes information or makes access causal. None creates data.

---

## 8. Out of scope

Named here so they are not mistaken for oversights:

- **Non-temporal duplicates** (Kapoor L1.4) — real leakage, but not time-series specific and not injectable into a price-series case.
- **Slippage, commission, and fill assumptions** — these make backtests optimistic but are *not* leakage; nothing about them violates the invariant. They belong in the "what remains unproven" section.
- **Overfitting through repeated parameter search** — real and serious, but a leak of the researcher's degrees of freedom rather than of future data. Disclosed, not detected.
- **Live-versus-backtest vendor differences** (restatements, point-in-time fundamentals) — out of reach without vendor data.

Stating these explicitly is worth points. A taxonomy claiming to catch everything invites the question it cannot answer.
