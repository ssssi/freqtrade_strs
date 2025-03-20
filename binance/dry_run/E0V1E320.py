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

fastk_dict = {}
cci_dict = {}

class E0V1E320(IStrategy):
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

    use_custom_stoploss = False

    is_optimize_32 = False
    buy_rsi_fast_32 = IntParameter(20, 70, default=40, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=42, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.973, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 1, default=0.69, decimals=2, space='buy', optimize=is_optimize_32)

    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=False)
    sell_time_threshold_1 = IntParameter(4, 24, default=7, space='sell', optimize=True)
    sell_loss_threshold_1 = DecimalParameter(-0.1, 0, default=-0.01, decimals=2, space='sell', optimize=True)
    sell_time_threshold_2 = IntParameter(8, 48, default=28, space='sell', optimize=True)
    sell_loss_threshold_2 = DecimalParameter(-0.2, 0, default=-0.16, decimals=2, space='sell', optimize=True)

    @property
    def protections(self):

        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 96
            }
        ]

    # def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #
    #     if current_profit >= 0.03:
    #         fastk_dict.pop(trade.id, None)
    #         cci_dict.pop(trade.id, None)
    #         return -0.002
    #
    #     return None
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
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)  # 新增波动率指标
        
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

        buy_new = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 34) &
                (dataframe['rsi'] > 28) &
                (dataframe['close'] < dataframe['sma_15'] * 0.96) &
                (dataframe['cti'] < self.buy_cti_32.value)
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
        hours_held = (current_time - trade.open_date_utc).total_seconds() / 3600

        atr_ratio = current_candle['atr'] / dataframe['atr'].mean()
        adjusted_loss_1 = self.sell_loss_threshold_1.value * atr_ratio
        adjusted_loss_2 = self.sell_loss_threshold_2.value * atr_ratio

        if current_profit > 0 and current_candle["fastk"] > self.sell_fastx.value:
            if not fastk_dict.get(trade.id):
                fastk_dict.update({trade.id: current_candle["close"]})

        if trade.id in fastk_dict and current_candle["close"] < fastk_dict.get(trade.id):
            fastk_dict.pop(trade.id, None)
            cci_dict.pop(trade.id, None)
            return "fastk_delay_sell"

        if trade.id in fastk_dict and current_profit < -0.02:
            fastk_dict.pop(trade.id, None)
            cci_dict.pop(trade.id, None)
            return "fastk_emergency_sell"

        if current_profit > -0.03 and current_candle["cci"] > 80:
            if not cci_dict.get(trade.id):
                cci_dict.update({trade.id: current_candle["close"]})

        if trade.id in cci_dict and current_candle["close"] < cci_dict.get(trade.id):
            fastk_dict.pop(trade.id, None)
            cci_dict.pop(trade.id, None)
            return "cci_loss_delay_sell"

        if trade.id in cci_dict and current_profit < -0.05:
            fastk_dict.pop(trade.id, None)
            cci_dict.pop(trade.id, None)
            return "cci_emergency_sell"

        # if current_time - timedelta(hours=7) > trade.open_date_utc:
        #     if current_profit >= -0.05:
        #         fastk_dict.pop(trade.id, None)
        #         cci_dict.pop(trade.id, None)
        #         return "time_loss_sell_7_5"
        #
        # if current_time - timedelta(hours=10) > trade.open_date_utc:
        #     if current_profit >= -0.1:
        #         fastk_dict.pop(trade.id, None)
        #         cci_dict.pop(trade.id, None)
        #         return "time_loss_sell_10_10"

        # 时间止损条件（带波动率调整和技术过滤）
        if hours_held > self.sell_time_threshold_1.value:
            if (current_profit >= adjusted_loss_1 and
                current_candle['rsi'] > 50):
                fastk_dict.pop(trade.id, None)
                cci_dict.pop(trade.id, None)
                return f"time_loss_sell_{self.sell_time_threshold_1.value}_{abs(adjusted_loss_1):.2f}"

        if hours_held > self.sell_time_threshold_2.value:
            if (current_profit >= adjusted_loss_2 or
                (hours_held > 24 and current_profit >= -0.15)):
                fastk_dict.pop(trade.id, None)
                cci_dict.pop(trade.id, None)
                return f"time_loss_sell_{self.sell_time_threshold_2.value}_{abs(adjusted_loss_2):.2f}"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe
