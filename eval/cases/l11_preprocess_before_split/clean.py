"""A forest on the four SPY features chosen on the training half."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    returns = close.pct_change()
    return pd.DataFrame(
        {
            "r1": returns,
            "r5": close.pct_change(5),
            "r10": close.pct_change(10),
            "r20": close.pct_change(20),
            "vol20": returns.rolling(20).std(),
            "vol60": returns.rolling(60).std(),
            "dist_sma": close / close.rolling(20).mean() - 1,
            "hl_range": (df["high"] - df["low"]) / close,
            "vol_ratio": returns.rolling(5).std() / returns.rolling(60).std(),
        }
    ).dropna()


def run_positions(df: pd.DataFrame) -> pd.Series:
    features = build_features(df)
    target = (df["close"].shift(-1) > df["close"]).astype(int).reindex(features.index)
    split = int(len(features) * 0.5)
    selector = SelectKBest(f_classif, k=4)
    selector.fit(features.iloc[:split], target.iloc[:split])
    chosen = features.loc[:, selector.get_support()]
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=0, n_jobs=1
    ).fit(chosen.iloc[:split], target.iloc[:split])
    evaluation = chosen.iloc[split:]
    pred = pd.Series(model.predict(evaluation), index=evaluation.index)
    signal = pred.reindex(df.index).fillna(0).astype(bool)
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
