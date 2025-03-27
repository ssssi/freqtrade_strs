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


class E0V1EN(IStrategy):
    minimal_roi = {
        "0": 1
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

    stoploss = -0.25
    trailing_stop = True
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    is_optimize_32 = False
    buy_rsi_fast_32 = IntParameter(20, 70, default=40, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=42, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.973, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 1, default=0.69, decimals=2, space='buy', optimize=is_optimize_32)


    # 新增可优化参数：24小时价格变化百分比范围
    buy_24h_min_pct = DecimalParameter(-30.0, 0.0, default=-15.0, decimals=1, space='buy', optimize=True)
    buy_24h_max_pct = DecimalParameter(0.0, 200.0, default=50.0, decimals=1, space='buy', optimize=True)

    buy_24h_min_pct1 = DecimalParameter(-30.0, 0.0, default=-15.0, decimals=1, space='buy', optimize=True)
    buy_24h_max_pct1 = DecimalParameter(0.0, 200.0, default=50.0, decimals=1, space='buy', optimize=True)



    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=True)

    @property
    def protections(self):

        return [
                {
                "method": "CooldownPeriod",
                "stop_duration_candles": 96
                }
               ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # buy_1 indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        dataframe['24h_change_pct'] = (dataframe['close'].pct_change(periods=288) * 100)


        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        buy_1 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                (dataframe['rsi'] > self.buy_rsi_32.value) &
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                (dataframe['cti'] < self.buy_cti_32.value) &
                (dataframe['24h_change_pct'] > self.buy_24h_min_pct1.value) &  # 使用可优化参数
                (dataframe['24h_change_pct'] < self.buy_24h_max_pct1.value)  # 使用可优化参数
        )

        buy_new = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 34) &
                (dataframe['rsi'] > 28) &
                (dataframe['close'] < dataframe['sma_15'] * 0.96) &
                (dataframe['cti'] < self.buy_cti_32.value) &
                (dataframe['24h_change_pct'] > self.buy_24h_min_pct.value) &  # 使用可优化参数
                (dataframe['24h_change_pct'] < self.buy_24h_max_pct.value)  # 使用可优化参数
        )


        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        conditions.append(buy_new)
        dataframe.loc[buy_new, 'enter_tag'] += 'buy_new'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_profit > 0 and "buy_new" == str(trade.enter_tag):
            if current_candle["fastk"] > self.sell_fastx.value:
                return "fastk_profit_sell"
        
        if current_profit > -0.03:
            if current_candle["cci"] > 80:
                return "cci_loss_sell"

        if current_time - timedelta(hours=7) > trade.open_date_utc:
            if current_profit >= -0.05:
                return "time_loss_sell_7_5"

        if current_time - timedelta(hours=10) > trade.open_date_utc:
            if current_profit >= -0.1:
                return "time_loss_sell_10_10"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe
