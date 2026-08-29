import pytest

from eval.inject import RECIPES, Injection, inject, repair

CLEAN = """import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    sma = df["close"].rolling(20).mean()
    signal = df["close"] > sma
    return signal.shift(1).fillna(False).astype(int)
"""


def test_inject_returns_the_leaked_source_and_its_ground_truth():
    result = inject(CLEAN, "L03")
    assert isinstance(result, Injection)
    assert result.leak_type == "L03"
    assert result.expected_inflation == "up"
    assert ".shift(1).fillna(False)" not in result.source
    assert ".fillna(False)" in result.source


def test_ground_truth_line_is_the_one_indexed_injected_line():
    result = inject(CLEAN, "L03")
    injected = result.source.splitlines()[result.line - 1]
    assert "fillna(False)" in injected
    assert result.line == 7


def test_repair_is_the_exact_inverse_of_inject():
    # Case validity rule 5 rests on this: the documented fix must reproduce the
    # clean source byte for byte, not merely lower the number.
    assert repair(inject(CLEAN, "L03").source, "L03") == CLEAN


def test_unknown_leak_type_is_rejected():
    with pytest.raises(ValueError, match="L99"):
        inject(CLEAN, "L99")


def test_missing_recipe_target_is_rejected():
    with pytest.raises(ValueError, match="not found"):
        inject("x = 1\n", "L03")


def test_ambiguous_recipe_target_is_rejected():
    # Two matches means the ground-truth line is not well defined.
    doubled = CLEAN + CLEAN
    with pytest.raises(ValueError, match="2 times"):
        inject(doubled, "L03")


def test_every_taxonomy_type_has_a_recipe():
    assert sorted(RECIPES) == [f"L{n:02d}" for n in range(1, 13)]


def test_every_recipe_changes_the_source():
    for leak_type, recipe in RECIPES.items():
        assert recipe.clean != recipe.leaked, leak_type
