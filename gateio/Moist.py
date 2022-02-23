# --- Do not remove these libs ---
from datetime import datetime

from freqtrade.strategy import IStrategy, DecimalParameter, stoploss_from_open
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import numpy as np


class Moist(IStrategy):
    """
        Moist
        modified from gettinMoist.
        https://github.com/werkkrew/freqtrade-strategies/blob/main/strategies/archived/gettinMoist.py
    """

    # Sell hyperspace params:
    sell_params = {
        # custom stoploss params, come from BB_RPB_TSL
        "pHSL": -0.99,
        "pPF_1": 0.02,
        "pPF_2": 0.05,
        "pSL_1": 0.02,
        "pSL_2": 0.04
    }

    minimal_roi = {
         "0": 1
    }

    # Stoploss:
    stoploss = -0.99

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Custom stoploss
    use_custom_stoploss = True

    startup_candle_count: int = 72
    process_only_new_candles = True

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
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['color'] = dataframe['close'] > dataframe['open']
    
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=6)

        dataframe['primed'] = np.where(dataframe['color'].rolling(3).sum() == 3, 1, 0)
        dataframe['in-the-mood'] = dataframe['rsi'] > dataframe['rsi'].rolling(12).mean()
        dataframe['macd_crossed_above'] = qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
        dataframe['macd_crossed_below'] = qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
        dataframe['throbbing'] = dataframe['roc'] > dataframe['roc'].rolling(12).mean()
        dataframe['ready-to-go'] = np.where(dataframe['close'] > dataframe['open'].rolling(12).mean(), 1, 0)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['primed']) &
                (dataframe['macd_crossed_above']) &
                (dataframe['throbbing']) &
                (dataframe['ready-to-go'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['macd_crossed_below']), 'sell'] = 1

        return dataframe
