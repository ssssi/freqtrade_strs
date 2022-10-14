# from datetime import datetime, timedelta

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce


# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    wr = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
    )

    return wr * -100


class E0V1E(IStrategy):
    minimal_roi = {
        "0": 100
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
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

    # Disabled
    stoploss = -0.1

    # Custom stoploss
    use_custom_stoploss = False

    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=46, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=19, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.942, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 0, default=-0.86, decimals=2, space='buy', optimize=is_optimize_32)

    is_optimize_r_deadfish = True
    buy_r_deadfish_ema = DecimalParameter(0.90, 1.2, default=1.087, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_bb_factor = DecimalParameter(0.90, 1.2, default=1.0, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_cti = DecimalParameter(-0.6, -0.0, default=-0.5, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_r14 = DecimalParameter(-60, -44, default=-60, space='buy', optimize=is_optimize_r_deadfish)

    sell_fastx = IntParameter(50, 100, default=75, space='sell', optimize=False)

    # # custom stoploss
    # trailing_optimize = True
    # sl_1 = DecimalParameter(0.001, 0.01, default=0.005, decimals=3, space='sell', optimize=trailing_optimize)
    # sl_2 = DecimalParameter(0.005, 0.02, default=0.01, decimals=3, space='sell', optimize=trailing_optimize)
    # sl_3 = DecimalParameter(0.008, 0.03, default=0.02, decimals=3, space='sell', optimize=trailing_optimize)
    # sl_4 = DecimalParameter(0.001, 0.008, default=0.002, decimals=3, space='sell', optimize=trailing_optimize)

    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #
    #     # evaluate highest to lowest, so that highest possible stop is used
    #     if current_profit > 0.08:
    #         return -self.sl_3.value
    #     elif current_profit > 0.05:
    #         return -self.sl_2.value
    #     elif current_profit >= 0.02:
    #         return -self.sl_1.value
    #
    #     if current_time - timedelta(minutes=60) > trade.open_date_utc:
    #         if 0.02 > current_profit >= 0.01:
    #             return -self.sl_4.value
    #
    #     return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # NFI 32 indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)

        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # dead fish indicators
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['bb_width'] = (
                (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        dataframe['r_14'] = williams_r(dataframe, period=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        buy_1 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                (dataframe['rsi'] > self.buy_rsi_32.value) &
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                (dataframe['cti'] < self.buy_cti_32.value)
        )

        buy_2 = (
                (dataframe['ema_100'] < dataframe['ema_200'] * self.buy_r_deadfish_ema.value) &
                (dataframe['bb_width'] > self.buy_r_deadfish_bb_width.value) &
                (dataframe['close'] < dataframe['bb_middleband2'] * self.buy_r_deadfish_bb_factor.value) &
                (dataframe['volume_mean_12'] > dataframe['volume_mean_24'] * self.buy_r_deadfish_volume_factor.value) &
                (dataframe['cti'] < self.buy_r_deadfish_cti.value) &
                (dataframe['r_14'] < self.buy_r_deadfish_r14.value)
        )

        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        conditions.append(buy_2)
        dataframe.loc[buy_2, 'enter_tag'] += 'buy_2'

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
        dataframe.loc[fastk_cross, 'exit_tag'] += 'fastk_cross'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'] = 1

        return dataframe
