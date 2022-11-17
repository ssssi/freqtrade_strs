from datetime import datetime, timedelta
from typing import Optional, Union
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce


def ewo(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


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

    is_optimize_ewo = True
    buy_rsi_fast = IntParameter(35, 50, default=45, space='buy', optimize=is_optimize_ewo)
    buy_rsi = IntParameter(15, 35, default=35, space='buy', optimize=is_optimize_ewo)
    buy_ewo = DecimalParameter(-6.0, 5, default=-5.585, space='buy', optimize=is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.942, space='buy', optimize=is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.084, space='buy', optimize=is_optimize_ewo)

    is_optimize_32 = False
    buy_rsi_fast_32 = IntParameter(20, 70, default=46, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=19, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.942, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 0, default=-0.86, decimals=2, space='buy', optimize=is_optimize_32)

    is_optimize_deadfish = False
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space='sell', optimize=is_optimize_deadfish)

    sell_fastx = IntParameter(50, 100, default=75, space='sell', optimize=False)
    delay_time = IntParameter(90, 1440, default=300, space='sell', optimize=True)

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

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['bb_width'] = (
                (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        # cofi & ewo
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        # Elliot
        dataframe['EWO'] = ewo(dataframe, 50, 200)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        is_ewo = (
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
                (dataframe['EWO'] > self.buy_ewo.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
                (dataframe['rsi'] < self.buy_rsi.value)
        )

        buy_1 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                (dataframe['rsi'] > self.buy_rsi_32.value) &
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                (dataframe['cti'] < self.buy_cti_32.value)
        )

        conditions.append(is_ewo)
        dataframe.loc[is_ewo, 'enter_tag'] += 'ewo'

        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        if current_time - timedelta(minutes=int(self.delay_time.value)) > trade.open_date_utc:
            if current_profit >= -0.01:
                return -0.005
         
        return self.stoploss

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "sell_fastk"

        # stoploss - deadfish
        if ((current_profit < self.sell_deadfish_profit.value)
                and (current_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (current_candle['close'] > current_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (current_candle['volume_mean_12'] < current_candle[
                    'volume_mean_24'] * self.sell_deadfish_volume_factor.value)):
            return "sell_stoploss_deadfish"

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')

        return dataframe
