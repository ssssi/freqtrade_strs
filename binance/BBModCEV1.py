from datetime import datetime, timedelta

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce


class BBModCEV1(IStrategy):

    minimal_roi = {
        "0": 100
    }

    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 20

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

    stoploss = -0.1

    # Custom stoploss
    use_custom_stoploss = True

    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=46, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=19, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.942, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 0, default=-0.86, decimals=2, space='buy', optimize=is_optimize_32)

    sell_fastx = IntParameter(50, 100, default=75, space='sell', optimize=False)

    # custom stoploss
    trailing_optimize = True
    sl_1 = DecimalParameter(0.001, 0.01, default=0.005, decimals=3, space='sell', optimize=trailing_optimize)
    sl_2 = DecimalParameter(0.005, 0.02, default=0.01, decimals=3, space='sell', optimize=trailing_optimize)
    sl_3 = DecimalParameter(0.008, 0.03, default=0.02, decimals=3, space='sell', optimize=trailing_optimize)
    sl_4 = DecimalParameter(0.001, 0.008, default=0.002, decimals=3, space='sell', optimize=trailing_optimize)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # evaluate highest to lowest, so that highest possible stop is used
        if current_profit > 0.08:
            return -self.sl_3.value
        elif current_profit > 0.05:
            return -self.sl_2.value
        elif current_profit >= 0.02:
            return -self.sl_1.value

        if current_time - timedelta(minutes=60) > trade.open_date_utc:
            if 0.02 > current_profit >= 0.01:
                return -self.sl_4.value

        return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)

        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        is_nfi_32 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                (dataframe['rsi'] > self.buy_rsi_32.value) &
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                (dataframe['cti'] < self.buy_cti_32.value)
        )

        conditions.append(is_nfi_32)
        dataframe.loc[is_nfi_32, 'enter_tag'] += 'nfi_32 '

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        fastk_cross = (
            (qtpylib.crossed_above(dataframe['fastk'], self.sell_fastx.value))
        )

        conditions.append(fastk_cross)
        dataframe.loc[fastk_cross, 'exit_tag'] += 'fastk_cross '

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'] = 1

        return dataframe
