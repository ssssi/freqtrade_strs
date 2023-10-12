from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce

TMP_HOLD = []


class E0V1E(IStrategy):
    minimal_roi = {
        "0": 10
    }

    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 120

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': True,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }

    stoploss = -0.18

    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=45, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=35, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.961, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 0, default=-0.58, decimals=2, space='buy', optimize=is_optimize_32)

    sell_fastx = IntParameter(50, 100, default=75, space='sell', optimize=True)

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

        if current_time - timedelta(minutes=90) < trade.open_date_utc:
            if current_profit >= 0.08:
                return "fastk_profit_sell_fast"

        if current_time - timedelta(hours=3) > trade.open_date_utc:
            if (current_candle["fastk"] >= 70) and (current_profit >= 0):
                return "fastk_profit_sell_delay"

        if current_time - timedelta(hours=4) > trade.open_date_utc:
            if current_profit > -0.06:
                return "fastk_loss_sell_delay"

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "fastk_profit_sell"

        if current_profit <= -0.1:
            # tmp hold
            if trade.id not in TMP_HOLD:
                TMP_HOLD.append(trade.id)

        for i in TMP_HOLD:
            # start recover sell it
            if trade.id == i and current_profit > -0.1:
                if current_candle["fastk"] > self.sell_fastx.value:
                    TMP_HOLD.remove(i)
                    return "fastk_loss_sell"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')

        return dataframe
