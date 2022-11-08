from datetime import datetime, timedelta
from typing import Optional, Union
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from freqtrade.strategy import DecimalParameter, IntParameter, informative
from functools import reduce

TMP_HOLD = []


# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    wr = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
    )

    return wr * -100


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

    is_optimize_cofi = True
    buy_roc_1h = IntParameter(-25, 200, default=10, space='buy', optimize=is_optimize_cofi)
    buy_bb_width_1h = DecimalParameter(0.3, 2.0, default=0.3, space='buy', optimize=is_optimize_cofi)
    buy_ema_cofi = DecimalParameter(0.94, 1.2, default=0.97, space='buy', optimize=is_optimize_cofi)
    buy_fastk = IntParameter(0, 40, default=20, space='buy', optimize=is_optimize_cofi)
    buy_fastd = IntParameter(0, 40, default=20, space='buy', optimize=is_optimize_cofi)
    buy_adx = IntParameter(0, 30, default=30, space='buy', optimize=is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, space='buy', optimize=is_optimize_cofi)
    buy_cofi_cti = DecimalParameter(-0.9, -0.0, default=-0.5, space='buy', optimize=is_optimize_cofi)
    buy_cofi_r14 = DecimalParameter(-100, -44, default=-60, space='buy', optimize=is_optimize_cofi)

    is_optimize_32 = False
    buy_rsi_fast_32 = IntParameter(20, 70, default=46, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=19, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.942, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 0, default=-0.86, decimals=2, space='buy', optimize=is_optimize_32)

    is_optimize_r_deadfish = False
    buy_r_deadfish_ema = DecimalParameter(0.90, 1.2, default=1.087, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_bb_factor = DecimalParameter(0.90, 1.2, default=1.0, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_cti = DecimalParameter(-0.6, -0.0, default=-0.5, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_r14 = DecimalParameter(-60, -44, default=-60, space='buy', optimize=is_optimize_r_deadfish)

    is_optimize_deadfish = False
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space='sell', optimize=is_optimize_deadfish)

    sell_fastx = IntParameter(50, 100, default=75, space='sell', optimize=False)

    # Time delay selling rule
    delay_time = IntParameter(90, 1440, default=360, space='sell', optimize=True)

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['roc'] = ta.ROC(dataframe, timeperiod=9)

        # # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = (
                (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        return dataframe

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

        # cofi & ewo
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['adx'] = ta.ADX(dataframe)
        # Elliot
        dataframe['EWO'] = ewo(dataframe, 50, 200)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        is_cofi = (
                (dataframe['roc_1h'] < self.buy_roc_1h.value) &
                (dataframe['bb_width_1h'] < self.buy_bb_width_1h.value) &
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value) &
                (dataframe['cti'] < self.buy_cofi_cti.value) &
                (dataframe['r_14'] < self.buy_cofi_r14.value)
        )

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

        buy_2 = (
                (dataframe['ema_100'] < dataframe['ema_200'] * self.buy_r_deadfish_ema.value) &
                (dataframe['bb_width'] > self.buy_r_deadfish_bb_width.value) &
                (dataframe['close'] < dataframe['bb_middleband2'] * self.buy_r_deadfish_bb_factor.value) &
                (dataframe['volume_mean_12'] > dataframe['volume_mean_24'] * self.buy_r_deadfish_volume_factor.value) &
                (dataframe['cti'] < self.buy_r_deadfish_cti.value) &
                (dataframe['r_14'] < self.buy_r_deadfish_r14.value)
        )

        conditions.append(is_cofi)
        dataframe.loc[is_cofi, 'enter_tag'] += 'cofi'

        conditions.append(is_ewo)
        dataframe.loc[is_ewo, 'enter_tag'] += 'ewo'

        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        conditions.append(buy_2)
        dataframe.loc[buy_2, 'enter_tag'] += 'buy_2'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        last_candle = dataframe.iloc[-2].squeeze()

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "sell_fastk"

        if current_time - timedelta(minutes=int(self.delay_time.value)) > trade.open_date_utc:
            if current_profit >= -0.01:
                #return "sell_delay_time"
                if trade.id not in TMP_HOLD:
                    TMP_HOLD.append(trade.id)
                    return None

        for i in TMP_HOLD:
            if trade.id == i and (current_candle["close"] < last_candle["close"]):
                TMP_HOLD.remove(i)
                return "sell_delay_time"

        # stoploss - deadfish
        if ((current_profit < self.sell_deadfish_profit.value)
                and (current_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (current_candle['close'] > current_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (current_candle['volume_mean_12'] < current_candle[
                    'volume_mean_24'] * self.sell_deadfish_volume_factor.value)
        ):
            return "sell_stoploss_deadfish"

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')

        return dataframe
