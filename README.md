# Hindsight

**Hindsight audits a backtest or time-series pipeline for look-ahead bias, and refuses to report any finding it has not proven by execution.**

Leak auditing for any time-series pipeline — demonstrated here on trading backtests, but the same bug class shows up in demand forecasting, predictive maintenance, credit risk, and medical prognosis: a model quietly reads data it would not have had at decision time.

## The bug, in one file

```python
def run_positions(df):
    sma = df["close"].rolling(20).mean()
    signal = df["close"] > sma      # <- uses today's close to decide today's trade
    return signal.fillna(False).astype(int)
```

This never crashes. It never fails a test. It just produces a better number than the strategy will ever produce live, because the decision at bar `t` is allowed to see the close of bar `t` — a price that does not exist yet at the moment the trade would actually be placed.

Run it through Hindsight:

```
$ hindsight audit golden_cross.py --data eval/data/SPY.csv --mode agent
```

```
verdict: leaks_proven
  L03 same-bar execution, line 8: signal = df["close"] > sma
  sharpe 3.72 -> 0.47   (before/after runs 2816524c7dde4e64 -> 151e59ef38ce4d18)
  fix:  signal = (df["close"] > sma).shift(1)
```

Sharpe 3.72 collapses to 0.47 the moment the strategy is denied the one bar of information it was never entitled to. Nothing was invented to get that number — `.shift(1)` only removes a row of access the strategy shouldn't have had. That run is real: `runs/20260831T142145-fd882229.json`, produced end-to-end through the CLI and reproduced identically through the web UI.

## The core mechanic

Hindsight never invents or predicts a value. Every patch it proposes is a *removal* of information — a `.shift(1)`, a lag, a masked row — never an addition. A candidate is scanned, patched, and the same strategy is re-run before and after with everything else held fixed. If the metric moves, that's a finding, backed by two persisted run IDs. If it doesn't move, or the patch doesn't apply, or the run crashes or produces zero trades, that's a *different*, distinctly-reported outcome — never silently folded into "clean."

This is enforced in code, not by convention: `Finding.__post_init__` cannot construct a finding without both run IDs, and `hindsight_core/hooks/verification.py` re-checks that both IDs point at a completed run in the store before a finding can reach a report. There is no code path that bypasses it.

## Where the agent actually is

Detecting "should this line be shifted" is one LLM call. The agent's job is judging **whether a number is believable yet, and what to try next if it isn't** — three places a straight scan-triage-patch line breaks down:

1. **The fix isn't always one line.** Higher-timeframe merge bugs need a repair constructed, run, and retried against the traceback.
2. **Fixing one leak reveals another.** A baseline can fall through several proven repairs before it's believable, and the agent has to keep going rather than stop at the first fix.
3. **A test isn't always conclusive.** "Leak proven," "the patch was wrong," and "no trades on this data" look identical from the outside and are reported as three separate outcomes, never collapsed into one.

Two detectors ship side by side specifically so this claim stays measurable rather than asserted: `hindsight_core/pipeline.py` walks the candidate list once and stops; `hindsight_core/orchestrator.py` retries, re-baselines after every kept repair, and re-opens candidates that were dismissed against a baseline that no longer exists. Measured on the 20 frozen eval cases:

| | pipeline (straight line) | agent (loop) |
|---|---|---|
| leaks detected | 9/12 | **11/12** |
| line localised | 7/12 | **8/12** |
| false positives | 0/8 | 0/8 |

Full arc, with numbers at every step, in `docs/CHANGELOG.md`.

## Architecture

```
                    hindsight_cli (CLI)          hindsight_server (FastAPI+SSE)
                            \                            /
                             \                          /
                              v                        v
                         hindsight_core   <- the only place logic lives
                    +-----------------------------------------------+
                    |  pipeline.py    (scan -> triage -> prove, 1 pass) |
                    |  orchestrator.py (agent loop: retry, re-baseline) |
                    |                                                 |
                    |  tools/  scan_file, apply_patch, run_backtest,  |
                    |          compare_runs, check_memory             |
                    |  provers/differential.py  (model retry, then a  |
                    |          mechanical sweep as the floor)         |
                    |  hooks/verification.py   BLOCKS unproven findings|
                    |  memory.py   JSON leak-signature store          |
                    |  sandbox.py  subprocess, timeout, seeded RNG    |
                    +-----------------------------------------------+
                              |
                              v
                   llm.py  Gemini -> Groq, cached, no third provider
```

The CLI and the server call the exact same `hindsight_core` functions. Nothing is duplicated between the two entry points — a behavior that exists on the web path but not the CLI path (or vice versa) is a bug.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env             # add GEMINI_API_KEY (and/or GROQ_API_KEY)

python -m eval.cache_data                          # pulls & caches 3y of daily bars
hindsight audit eval/cases/l03_same_bar_execution/strategy.py \
    --data eval/data/SPY.csv --mode agent
```

Full setup, the CLI, the server, and the web UI are all in `docs/REPRODUCE.md`, with every command run before it was written down.

## Evaluation

Twenty frozen cases — twelve injected leaks (one per taxonomy type) and eight clean controls that must not fire — plus case 21, added after freeze: a native reproduction of [freqtrade issue #11346](https://github.com/freqtrade/freqtrade/issues/11346), a real bug that freqtrade's own `lookahead-analysis` tool reports as clean.

```bash
hindsight eval --detector agent --repeat 3 --json --out eval/results/agent.json
hindsight eval --case htf_merge_11346 --detector agent
```

Case 21 head-to-head, same file, same window, same data:

| | verdict | Sharpe |
|---|---|---|
| **Hindsight agent** | leak proven by execution | 3.589 → **0.461** |
| **freqtrade lookahead-analysis** | `has_bias: No`, `biased_indicators` empty | — |

## What Hindsight will not do

It will not suggest strategy improvements. Advice can't be execution-proven, and a tool whose entire pitch is "prove everything" doesn't get to ship an unprovable claim on the side. It will not invent a value to force a patch through — if the fix needs a substitute column rather than a removal, it reports "detected, not mechanically patchable" instead of guessing.

## What remains unproven

Zero slippage and zero commission are assumed throughout. The eight-ETF universe the demo data is cached from exists today, which is a survivorship bias the tool does not correct for. Every eval case is optimised and tested on the same window. The sandbox seeds the RNG so a before/after delta is attributable to the patch and not to the draw — which also means an ML-based strategy's result is conditional on that one seed, not an absolute. Full accounting, with numbers, in `docs/CHANGELOG.md`.

## Hot take

On a silent bug, an agent's confidence is worthless — a broken prover that reports "clean" on every input looks exactly like a working one. So don't trust the model's judgment; force falsification by execution, and structurally forbid any claim from reaching the report that doesn't carry the run IDs to back it up.

## License

MIT — see `LICENSE`.
