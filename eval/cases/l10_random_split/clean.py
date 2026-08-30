"""A random forest on SPY price features, trained on the earlier half."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
    x_train, _, y_train, _ = train_test_split(
        features, target, test_size=0.5, shuffle=False, random_state=0
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=1, random_state=0, n_jobs=1
    ).fit(x_train, y_train)
    evaluation = features.iloc[len(features) // 2 :]
    pred = pd.Series(model.predict(evaluation), index=evaluation.index)
    signal = pred.reindex(df.index).fillna(0).astype(bool)
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
