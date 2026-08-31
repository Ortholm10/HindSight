"""Freqtrade issue #11346, as a native freqtrade strategy.

Not the generic shim in eval/baselines/freqtrade.py. That shim resamples the
higher timeframe itself and hands freqtrade one pre-merged frame, so it never
exercises freqtrade's own informative machinery - which is the whole point
here. This strategy declares informative_pairs() and pulls the weekly frame
through self.dp.get_pair_dataframe(), exactly as a freqtrade user would.

The leak is the merge, and it is the reporter's own line:

    matching_indices = pair_dataframe[pair_dataframe['date'] <= row_time].index

Their diagnosis, verbatim: "In a live run or dry run, if it's currently 14:15
pm, we can get a 15m candle for 14:00, but we have to wait until exactly 15:00
to get a 1h candle for 14:00." Here the same mistake at 1d/1w: Monday reads a
weekly candle that does not close until Friday.

ft_clean.py is this file with one change - freqtrade's own
merge_informative_pair() in place of the hand-rolled merge.
"""
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy


class Leak11346(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1d"
    inf_timeframe = "1w"
    minimal_roi = {"0": 100}
    stoploss = -0.99
    startup_candle_count = 40
    use_exit_signal = True
    can_short = False
    process_only_new_candles = False

    def informative_pairs(self):
        return [(pair, self.inf_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        inf = self.dp.get_pair_dataframe(metadata["pair"], self.inf_timeframe)
        inf = inf.copy()
        inf["w_sma"] = inf["close"].rolling(4).mean()
        inf["w_up"] = (inf["close"] > inf["w_sma"]).astype(int)
        # #11346, vectorised: merge_asof(direction="backward") selects the
        # latest informative row whose OPEN date is <= this row's date -
        # identical semantics to the reporter's date <= row_time filter, and
        # identically wrong, because that row is the week still in progress.
        merged = pd.merge_asof(
            dataframe, inf[["date", "w_up"]], on="date", direction="backward"
        )
        dataframe["w_up"] = merged["w_up"].ffill().fillna(0).to_numpy()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        prior = dataframe["w_up"].shift(1).fillna(0)
        dataframe.loc[(dataframe["w_up"] == 1) & (prior != 1), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        prior = dataframe["w_up"].shift(1).fillna(0)
        dataframe.loc[(dataframe["w_up"] == 0) & (prior == 1), "exit_long"] = 1
        return dataframe
