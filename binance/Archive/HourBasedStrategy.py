from freqtrade.strategy import IntParameter, IStrategy
from pandas import DataFrame


class HourBasedStrategy_5m(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        "buy_hour_max": 24,
        "buy_hour_min": 4,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_hour_max": 21,
        "sell_hour_min": 22,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.528,
        "169": 0.113,
        "528": 0.089,
        "1837": 0
    }

    # Stoploss:
    stoploss = -0.10

    # Optimal timeframe
    timeframe = '5m'

    buy_hour_min = IntParameter(0, 1440, default=1, space='buy')
    buy_hour_max = IntParameter(0, 1440, default=0, space='buy')

    sell_hour_min = IntParameter(0, 1440, default=1, space='sell')
    sell_hour_max = IntParameter(0, 1440, default=0, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['minute'] = dataframe['date'].dt.minute
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['minute'].between(self.buy_hour_min.value, self.buy_hour_max.value))
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['minute'].between(self.sell_hour_min.value, self.sell_hour_max.value))
            ),
            'exit_long'] = 1
        return dataframe
