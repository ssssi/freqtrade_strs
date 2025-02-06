from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)


class E0V1E(IStrategy):
    minimal_roi = {
        "0": 1
    }
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 240
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

    stoploss = -0.25
    trailing_stop = True
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    use_custom_stoploss = False

    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=40, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=42, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.973, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 1, default=0.69, decimals=2, space='buy', optimize=is_optimize_32)

    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=True)

    cci_opt = False
    sell_loss_cci = IntParameter(low=0, high=600, default=140, space='sell', optimize=cci_opt)
    sell_loss_cci_profit = DecimalParameter(-0.15, 0, default=-0.05, decimals=2, space='sell', optimize=cci_opt)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # buy_1 indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)

        dataframe['ma120'] = ta.MA(dataframe, timeperiod=120)
        dataframe['ma240'] = ta.MA(dataframe, timeperiod=240)

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

        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        min_profit = trade.calc_profit_ratio(trade.min_rate)
        
        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "fastk_profit_sell"
        
        if min_profit <= -0.15:
            if current_profit > self.sell_loss_cci_profit.value:
                if current_candle["cci"] > self.sell_loss_cci.value:
                    return "cci_loss_sell"

        if current_profit >= -0.03:
            if current_candle["cci"] > 80:
                return "cci_loss_sell_1"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe
