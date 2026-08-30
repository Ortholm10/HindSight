"""A scaler fitted on the whole frame, with no `split` name anywhere in scope.

`fit_in_fold` rewrites `.fit(X)` to `.fit(X.iloc[:split])`, and `split` does not
exist here — so the repair applies cleanly, and then the patched strategy dies
with a NameError. That traceback is what the prover has to read and route
around. It is a real consequence of the closed vocabulary meeting real code,
not a staged failure: apply_patch's own docstring says the fold boundary is
named and never computed, precisely so a missing name surfaces as a crash
rather than as an invented number.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def run_positions(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    features = pd.DataFrame({"ret": close.pct_change().fillna(0.0)})
    scaler = StandardScaler()
    scaler.fit(features)
    scaled = pd.Series(scaler.transform(features)[:, 0], index=close.index)
    signal = scaled > 0
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
