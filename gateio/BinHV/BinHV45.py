# --- Do not remove these libs ---
from datetime import datetime

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, DecimalParameter, stoploss_from_open
from freqtrade.strategy import IntParameter
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)

    return rolling_mean, lower_band


class BinHV45(IStrategy):

    minimal_roi = {
        "0": 100
    }

    timeframe = '1m'

    buy_bbdelta = IntParameter(low=1, high=15, default=30, space='buy', optimize=True)
    buy_closedelta = IntParameter(low=15, high=20, default=30, space='buy', optimize=True)
    buy_tail = IntParameter(low=20, high=30, default=30, space='buy', optimize=True)

    # Hyperopt parameters
    buy_params = {
        "buy_bbdelta": 7,
        "buy_closedelta": 17,
        "buy_tail": 25,
    }

    sell_params = {
        # custom stoploss params, come from BB_RPB_TSL
        "pHSL": -0.05,
        "pPF_1": 0.0125,
        "pPF_2": 0.05,
        "pSL_1": 0.008,
        "pSL_2": 0.04
    }

    # Stoploss:
    stoploss = -0.99  # use custom stoploss

    # Custom stoploss
    use_custom_stoploss = True

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'emergencysell': 'limit',
        'forcebuy': "limit",
        'forcesell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # hard stoploss profit
    pHSL = DecimalParameter(-0.990, -0.040, default=-0.08, decimals=3, space='sell')
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.050, default=0.016, decimals=3, space='sell')
    pSL_1 = DecimalParameter(0.008, 0.050, default=0.011, decimals=3, space='sell')

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell')
    pSL_2 = DecimalParameter(0.040, 0.100, default=0.040, decimals=3, space='sell')

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        hsl = self.pHSL.value
        pf_1 = self.pPF_1.value
        sl_1 = self.pSL_1.value
        pf_2 = self.pPF_2.value
        sl_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > pf_2:
            sl_profit = sl_2 + (current_profit - pf_2)
        elif current_profit > pf_1:
            sl_profit = sl_1 + ((current_profit - pf_1) * (sl_2 - sl_1) / (pf_2 - pf_1))
        else:
            sl_profit = hsl

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)

        dataframe['upper'] = bollinger['upper']
        dataframe['mid'] = bollinger['mid']
        dataframe['lower'] = bollinger['lower']
        dataframe['bbdelta'] = (dataframe['mid'] - dataframe['lower']).abs()
        dataframe['pricedelta'] = (dataframe['open'] - dataframe['close']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['lower'].shift().gt(0) &
                dataframe['bbdelta'].gt(dataframe['close'] * self.buy_bbdelta.value / 1000) &
                dataframe['closedelta'].gt(dataframe['close'] * self.buy_closedelta.value / 1000) &
                dataframe['tail'].lt(dataframe['bbdelta'] * self.buy_tail.value / 1000) &
                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift())
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        no sell signal
        """
        dataframe.loc[:, 'sell'] = 0
        return dataframe
