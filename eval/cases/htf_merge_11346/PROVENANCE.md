# Case 21 — freqtrade issue #11346

**We did not design this case.** That is the whole point of it. It is
transcribed from a bug a freqtrade user reported against freqtrade's own
look-ahead detector, and the detector still passes it.

Source: <https://github.com/freqtrade/freqtrade/issues/11346> (opened 2025-02-07,
freqtrade 2024.11-dev, still open at the time of writing).

## What the reporter found

> I have a strategy now, which definitely uses future data because its
> backtesting results are so good while live performance is exactly the
> opposite. Strangely, I conducted both lookahead analysis and recursive
> analysis on it, and the results showed no bias.

Their own diagnosis, naming their own line:

> The problem lies in the code
> `matching_indices = pair_dataframe[pair_dataframe['date'] <= row_time].index`.
> In a live run or dry run, if it's currently 14:15 pm, we can get a 15m candle
> for 14:00, but we have to wait until exactly 15:00 to get a 1h candle for
> 14:00. Now this code has made a mistake in backtesting, as it allows the
> program to access the future data of the 1h candle for 14:00 when it is 14:15.

And the tool's verdict on it: `has_bias = No`, `total_signals = 20`,
`biased_entry_signals = 0`, `biased_exit_signals = 0`, `biased_indicators`
empty.

## What we changed, and why

| Their setup | Ours | Why |
|---|---|---|
| 15m base, 1h informative | 1d base, 1w informative | This eval caches three years of daily SPY bars and nothing else (CLAUDE.md rule 5). The bug is about a *ratio* of timeframes, not about minutes. |
| A market-mode classifier over N consecutive closes | Weekly close vs weekly SMA(4) | Their signal logic is incidental; the merge is the bug. A smaller signal keeps the case readable and keeps the leak the only difference between `strategy.py` and `clean.py`. |
| A `for`-loop `date <= row_time` filter | `merge_asof(direction="backward")` | Identical semantics — both select the latest informative row whose OPEN date is at or before the current row — without an O(n²) loop. |

Everything that makes the bug a bug is preserved: an informative series stamped
at its open, joined to base rows that fall *inside* it.

**One correction to the framing this case was requested under.** The reporter
did **not** use `@informative` or `merge_informative_pair`. Their strategy has
no `informative_pairs` method at all; they hand-rolled the merge. That is
precisely the bug — `merge_informative_pair` is *safe*, and its docstring says
so ("Moves the date of the informative pair by 1 time interval forward"). So the
faithful reproduction cannot put the leak inside that helper. What
`ft_strategy.py` does instead is use the native informative path for everything
except the merge — `informative_pairs()` and `self.dp.get_pair_dataframe()` —
and then bypass the helper exactly as the reporter did. `ft_clean.py` is the
same file with `merge_informative_pair()` restored, which is the fix.

## The four files

| File | What it is |
|---|---|
| `strategy.py` | The leak in pandas. What Hindsight audits. |
| `clean.py` | The repair: the weekly series lagged one full weekly period. |
| `ft_strategy.py` | The leak as a native freqtrade `IStrategy`. What freqtrade audits. |
| `ft_clean.py` | The repair, via `merge_informative_pair()`. |

## Measured

Hindsight's own runner (`eval/runner.py`, three years of daily SPY):

| variant | Sharpe | total return | max drawdown | position changes |
|---|---|---|---|---|
| `strategy.py` (leaked) | **3.589** | 2.258 | −0.046 | 42 |
| `clean.py` (repaired) | **0.461** | 0.154 | −0.202 | 42 |

`freqtrade backtesting`, same window, on the native strategies:

| variant | trades | avg profit | total profit | win rate |
|---|---|---|---|---|
| `Leak11346` | 20 | 5.54% | 1.11% | **95.0%** |
| `Clean11346` | 20 | 0.73% | 0.15% | **50.0%** |

Identical trade counts, identical durations. The leak does not trade more; it
trades *right*.

`freqtrade lookahead-analysis`, same window, `--minimum-trade-amount 5
--targeted-trade-amount 20`:

```
| filename     | strategy  | has_bias | total_signals | biased_entry_signals | biased_exit_signals | biased_indicators |
| Leak11346.py | Leak11346 |       No |            20 |                    0 |                   0 |                   |
```

A strategy that wins 95% of its trades by reading the current week's close,
and the official tool reports **no bias** — the same shape of table the
reporter posted.

## Why the tool misses it

Read from freqtrade's installed source, not inferred:

1. `freqtrade/optimize/analysis/lookahead.py:146-148` truncates each re-run to
   the trade's entry candle **plus exactly one base candle**, and hardcodes that
   window with no config key.
2. `freqtrade/data/dataprovider.py::historic_ohlcv` loads the *informative*
   series against that same truncated timerange, filtering candles by their
   **open** date.

So a weekly candle stamped Monday survives a truncation to Wednesday — and it
survives it **whole**, because the stored candle already contains Friday's
close. The full-history run and the truncated run read the identical value.
Nothing differs, so nothing is reported.

The blindness is structural, not a tuning problem: no truncation on open-date
boundaries can expose a leak that lives *inside* a single higher-timeframe
candle. Widening the grace window would not help, because the leak is one
candle deep at the informative resolution.

## A session-2 question this case settles

`eval/baselines/freqtrade.py` documents an artifact where freqtrade's raw
`has_bias` fires `True` on nearly every case, clean ones included, and
concludes `biased_indicators` is the more trustworthy signal.

That artifact is **shim-specific**. On these native strategies freqtrade
reports `has_bias = False` for both the leaked and the repaired variant — no
timing noise at all. It came from the shim converting a 0/1 position series
into discrete enter/exit events, which interacts with next-candle market-order
fill timing. It is not a property of freqtrade's analyzer.

`biased_indicators` was used as the cross-check here anyway, and it is empty
for both variants. Both signals agree, and both are wrong.
