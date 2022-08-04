# --- Do not remove these libs ---
from functools import reduce

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from datetime import datetime
# --------------------------------
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, stoploss_from_open
from pandas import DataFrame


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


class CombinedBinHAndCluc(IStrategy):
    minimal_roi = {
        "0": 0.05
    }
    stoploss = -0.99
    timeframe = '5m'

    process_only_new_candles = True

    # Custom stoploss
    use_custom_stoploss = True

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }

    bhv45_op = True
    buy_bbdelta = DecimalParameter(0.001, 0.015, default=0.008, decimals=3, space='buy', optimize=bhv45_op)
    buy_closedelta = DecimalParameter(0.0150, 0.0200, default=0.0175, decimals=4, space='buy', optimize=bhv45_op)
    buy_tail = DecimalParameter(0.200, 0.300, default=0.25, decimals=3, space='buy', optimize=bhv45_op)

    cm8_op = True
    buy_low = DecimalParameter(0.900, 1, default=0.985, decimals=3, space='buy', optimize=cm8_op)

    leverage_optimize = False
    leverage_num = IntParameter(low=1, high=3, default=3, space='buy', optimize=leverage_optimize)

    # custom stoploss
    trailing_optimize = True
    pHSL = DecimalParameter(-0.990, -0.040, default=-0.15, decimals=3, space='sell', optimize=True)
    pPF_1 = DecimalParameter(0.008, 0.100, default=0.03, decimals=3, space='sell', optimize=False)
    pSL_1 = DecimalParameter(0.01, 0.030, default=0.025, decimals=3, space='sell', optimize=trailing_optimize)
    pPF_2 = DecimalParameter(0.040, 0.200, default=0.10, decimals=3, space='sell', optimize=False)
    pSL_2 = DecimalParameter(0.040, 0.100, default=0.09, decimals=3, space='sell', optimize=trailing_optimize)

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
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

        if self.can_short:
            if (-1 + ((1 - sl_profit) / (1 - current_profit))) <= 0:
                return 1
        else:
            if (1 - ((1 + sl_profit) / (1 + current_profit))) <= 0:
                return 1

        return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # strategy BinHV45
        mid, lower = bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        # strategy ClucMay72018
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        bhv45 = (  # strategy BinHV45
                dataframe['lower'].shift().gt(0) &
                dataframe['bbdelta'].gt(dataframe['close'] * self.buy_bbdelta.value) &
                dataframe['closedelta'].gt(dataframe['close'] * self.buy_closedelta.value) &
                dataframe['tail'].lt(dataframe['bbdelta'] * self.buy_tail.value) &
                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift())
        )

        cm8 = (  # strategy ClucMay72018
                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < self.buy_low.value * dataframe['bb_lowerband']) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 20))
        )

        conditions.append(bhv45)
        dataframe.loc[bhv45, 'enter_tag'] += 'bhv45 '

        conditions.append(cm8)
        dataframe.loc[cm8, 'enter_tag'] += 'cm8 '

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[
            (qtpylib.crossed_above(dataframe['close'], dataframe['bb_middleband'])),
            'exit_long'
        ] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value
