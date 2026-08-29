"""A logistic classifier on SPY features, cross-validated chronologically.

Traps L10 and L11. The split is TimeSeriesSplit, and the scaler sits inside the
Pipeline so it is refitted per fold. The final training row of each fold is
dropped because its label reads the first test bar.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    returns = close.pct_change()
    return pd.DataFrame(
        {
            "r1": returns,
            "r5": close.pct_change(5),
            "vol20": returns.rolling(20).std(),
            "dist_sma": close / close.rolling(20).mean() - 1,
        }
    ).dropna()


def run_positions(df: pd.DataFrame) -> pd.Series:
    features = build_features(df)
    target = (df["close"].shift(-1) > df["close"]).astype(int).reindex(features.index)
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    pred = pd.Series(0, index=features.index)
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(features):
        # Embargo the last training row: its label reads the first test bar.
        fit_idx = train_idx[:-1]
        pipeline.fit(features.iloc[fit_idx], target.iloc[fit_idx])
        pred.iloc[test_idx] = pipeline.predict(features.iloc[test_idx])
    signal = pred.reindex(df.index).fillna(0).astype(bool)
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
