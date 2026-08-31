"""Case 21 repaired, using freqtrade's own merge_informative_pair().

The helper stamps each informative row at open + one interval - 1 base candle,
so the weekly bar becomes visible only after it has closed. Nothing else about
the strategy changes, which is what makes the before/after comparable.
"""
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy, merge_informative_pair


class Clean11346(IStrategy):
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
        # The one line that separates this file from ft_strategy.py.
        dataframe = merge_informative_pair(
            dataframe, inf[["date", "w_up"]], self.timeframe, self.inf_timeframe
        )
        dataframe["w_up"] = dataframe[f"w_up_{self.inf_timeframe}"].fillna(0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        prior = dataframe["w_up"].shift(1).fillna(0)
        dataframe.loc[(dataframe["w_up"] == 1) & (prior != 1), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        prior = dataframe["w_up"].shift(1).fillna(0)
        dataframe.loc[(dataframe["w_up"] == 0) & (prior == 1), "exit_long"] = 1
        return dataframe
