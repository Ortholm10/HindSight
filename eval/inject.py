"""Leak injection, implemented literally from docs/taxonomy.md section 4.

Every recipe is a single-fragment textual substitution, and that is deliberate.
It buys three properties the eval depends on:

* the ground-truth line is exact, because exactly one fragment moved;
* the documented fix is the *exact* inverse of the injection, which is what
  case validity rule 5 asserts;
* clean and leaked differ by nothing else, so a Sharpe difference between them
  can only be the leak.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    clean: str
    leaked: str
    expected_inflation: str = "up"


@dataclass(frozen=True)
class Injection:
    source: str
    leak_type: str
    line: int
    expected_inflation: str


# Keyed by taxonomy ID. `clean` must appear exactly once in the clean case.
RECIPES: dict[str, Recipe] = {
    # Name a future row directly.
    "L01": Recipe(
        clean='signal = df["close"] > sma',
        leaked='signal = df["close"].shift(-1) > sma',
    ),
    # Label the window at its midpoint, pulling ~n/2 future bars into every value.
    "L02": Recipe(
        clean='sma = df["close"].rolling(60).mean()',
        leaked='sma = df["close"].rolling(60, center=True).mean()',
    ),
    # Drop the lag between deriving a signal and holding it.
    "L03": Recipe(
        clean=".shift(1).fillna(False)",
        leaked=".fillna(False)",
    ),
    # Merge the higher-timeframe series without waiting for its bar to close.
    # merge_asof(direction="backward") finds the currently-open weekly bar, whose
    # mean already reflects data through t. Only the pre-merge lag makes it safe.
    "L04": Recipe(
        clean="htf_values = weekly_sma.shift(1)",
        leaked="htf_values = weekly_sma",
    ),
    # Scale by a statistic that already knows the whole history.
    "L05": Recipe(
        clean="cutoff = spread.rolling(40, min_periods=20).quantile(0.15)",
        leaked="cutoff = spread.quantile(0.15)",
    ),
    # Fill a gap with the next known value instead of the last.
    "L06": Recipe(
        clean="macro = macro.ffill()",
        leaked="macro = macro.bfill()",
    ),
    # Stamp a completed monthly aggregate at the moment its month began.
    # "MS" is one of the frequencies pandas labels LEFT by default, so the bare
    # call hands every day of a month that whole month's mean. "W" and "ME"
    # label right; that inconsistency is the trap, not a detail.
    "L07": Recipe(
        clean=(
            'monthly = closes.resample("MS", label="right", closed="right")'
            ".mean().shift(1)"
        ),
        leaked='monthly = closes.resample("MS").mean()',
    ),
    # Promote a forward-looking reporting column into the decision.
    "L08": Recipe(
        clean="signal = momentum > 0",
        leaked='signal = forward_max > df["close"] * 1.05',
    ),
    # Mark a peak at its own timestamp instead of after it is confirmed.
    "L09": Recipe(
        clean="troughs = causal_troughs(close, 10)",
        leaked="troughs = two_sided_troughs(close, 10)",
    ),
    # Interleave test rows with training rows minutes apart.
    "L10": Recipe(
        clean="shuffle=False",
        leaked="shuffle=True",
    ),
    # Choose the features using data the training fold is not entitled to.
    "L11": Recipe(
        clean="selector.fit(features.iloc[:split], target.iloc[:split])",
        leaked="selector.fit(features, target)",
    ),
    # Choose the instruments using the end of the period.
    "L12": Recipe(
        clean="universe = SYMBOLS",
        leaked="universe = [s for s in SYMBOLS if _survived(closes[s])]",
    ),
}


def _substitute(source: str, old: str, new: str, what: str) -> tuple[str, int]:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"{what} target not found in source: {old!r}")
    if count != 1:
        raise ValueError(
            f"{what} target is ambiguous: {old!r} appears {count} times, "
            "so the ground-truth line is not well defined"
        )
    index = source.index(old)
    line = source.count("\n", 0, index) + 1
    return source[:index] + new + source[index + len(old) :], line


def _recipe(leak_type: str) -> Recipe:
    try:
        return RECIPES[leak_type]
    except KeyError:
        raise ValueError(
            f"unknown leak type {leak_type!r}; docs/taxonomy.md defines "
            f"{', '.join(sorted(RECIPES))}"
        ) from None


def inject(clean_source: str, leak_type: str) -> Injection:
    recipe = _recipe(leak_type)
    source, line = _substitute(clean_source, recipe.clean, recipe.leaked, "injection")
    return Injection(
        source=source,
        leak_type=leak_type,
        line=line,
        expected_inflation=recipe.expected_inflation,
    )


def repair(leaked_source: str, leak_type: str) -> str:
    """The documented fix. Exact inverse of `inject` by construction."""
    recipe = _recipe(leak_type)
    source, _ = _substitute(leaked_source, recipe.leaked, recipe.clean, "repair")
    return source
