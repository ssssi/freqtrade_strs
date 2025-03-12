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
    trailing_stop = False
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    use_custom_stoploss = True

    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=40, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=42, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.973, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 1, default=0.69, decimals=2, space='buy', optimize=is_optimize_32)

    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=True)
    sell_fastk_retracement = DecimalParameter(0.95, 1.0, default=0.98, decimals=2, space='sell', optimize=True)
    max_allowed_drawdown = DecimalParameter(-0.05, -0.01, default=-0.05, decimals=2, space='sell', optimize=True)
    min_redline_pct = DecimalParameter(0.005, 0.03, default=0.01, decimals=3, space='sell', optimize=True)

    fastk_states = {}
    cci_states = {}

    @property
    def protections(self):

        return [
                {
                "method": "CooldownPeriod",
                "stop_duration_candles": 96
                }
               ]

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if current_profit >= 0.05:
            return -0.003

        if not self.fastk_states.get(pair, {}).get('triggered', False):
            if "buy_new" in str(trade.enter_tag) and current_profit >= 0.03:
                return -0.002

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # 清除已平仓交易对的状态

        active_pairs = [metadata['pair']]
        for pair in list(self.fastk_states.keys()):
            if pair not in active_pairs:
                del self.fastk_states[pair]

        for pair in list(self.cci_states.keys()):
            if pair not in active_pairs:
                del self.cci_states[pair]
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

        if pair not in self.fastk_states:
            self.fastk_states[pair] = {
                'triggered': False,
                'peak_value': 0
            }

        if pair not in self.cci_states:
            self.cci_states[pair] = {
                'cci_triggered': False,
                'trigger_candle': None,
                'consecutive_red': 0
            }

        if current_profit > 0:
            # 当首次触发条件时记录峰值

            if current_candle["fastk"] > self.sell_fastx.value:

                if not self.fastk_states[pair]['triggered']:

                    self.fastk_states[pair]['peak_value'] = current_candle["fastk"]
                    self.fastk_states[pair]['triggered'] = True
                # 已触发条件后检查下降
                elif self.fastk_states[pair]['triggered']:
                    if current_candle["fastk"] < (
                            self.fastk_states[pair]['peak_value'] * self.sell_fastk_retracement.value):
                        self.fastk_states[pair] = {'triggered': False, 'peak_value': 0}  # 重置状态

                        return "fastk_profit_delay_sell"

        if current_profit > -0.03:
            if current_candle["cci"] > 80:

                if not self.cci_states[pair]['cci_triggered']:
                    self.cci_states[pair] = {
                        'cci_triggered': True,
                        'trigger_candle': current_candle.copy(),
                        'consecutive_red': 0
                    }
                if self.cci_states[pair]['cci_triggered']:

                    is_red_candle = current_candle['close'] < current_candle['open']

                    # 更新连续阴线计数
                    if is_red_candle:
                        self.cci_states[pair]['consecutive_red'] += 1
                    else:
                        self.cci_states[pair]['consecutive_red'] = 0  # 阳线则重置

                    open_price = current_candle['open']
                    close_price = current_candle['close']
                    price_drop_pct = (open_price - close_price) / open_price

                    sell_condition = (
                        # 条件1: 单根阴线跌幅达标
                            (price_drop_pct >= self.min_redline_pct.value) |
                            # 条件2: 连续两根阴线（无论跌幅）
                            (self.cci_states[pair]['consecutive_red'] >= 2)
                    )


                    if is_red_candle and sell_condition:
                        self.cci_states[pair] = {'cci_triggered': False, 'trigger_candle': None, 'consecutive_red': 0}

                        return "cci_loss_sell_delay"  # 修改退出标签以区分

                    if current_profit < self.max_allowed_drawdown.value:
                        self.cci_states[pair] = {'cci_triggered': False, 'trigger_candle': None, 'consecutive_red': 0}
                        return "cci_emergency_sell"

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
