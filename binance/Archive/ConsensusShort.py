from datetime import datetime

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series

import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, DecimalParameter,  CategoricalParameter
from freqtrade.strategy.interface import IStrategy
from technical.consensus import Consensus


# import freqtrade.vendor.qtpylib.indicators as qtpylib

# custom indicators
# ##################################################################################################
def RMI(dataframe, *, length=20, mom=5):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
    """
    df = dataframe.copy()

    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]


def zema(dataframe, period, field='close'):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/overlap_studies.py#L79
    Modified slightly to use ta.EMA instead of technical ema
    """
    df = dataframe.copy()

    df['ema1'] = ta.EMA(df[field], timeperiod=period)
    df['ema2'] = ta.EMA(df['ema1'], timeperiod=period)
    df['d'] = df['ema1'] - df['ema2']
    df['zema'] = df['ema1'] + df['d']

    return df['zema']


def same_length(bigger, shorter):
    return np.concatenate((np.full((bigger.shape[0] - shorter.shape[0]), np.nan), shorter))


def mastreak(dataframe: DataFrame, period: int = 4, field='close') -> Series:
    """
    MA Streak
    Port of: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
    """
    df = dataframe.copy()

    avgval = zema(df, period, field)

    arr = np.diff(avgval)
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                      -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))

    res = same_length(df['close'], streak)

    return res


def linear_growth(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
    """
    Simple linear growth function. Grows from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (end - start) / (end_time - start_time)

    return min(end, start + (rate * time))


def pcc(dataframe: DataFrame, period: int = 20, mult: int = 2):
    """
    Percent Change Channel
    PCC is like KC unless it uses percentage changes in price to set channel distance.
    https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
    """
    df = dataframe.copy()

    df['previous_close'] = df['close'].shift()

    df['close_change'] = (df['close'] - df['previous_close']) / df['previous_close'] * 100
    df['high_change'] = (df['high'] - df['close']) / df['close'] * 100
    df['low_change'] = (df['low'] - df['close']) / df['close'] * 100

    df['delta'] = df['high_change'] - df['low_change']

    mid = zema(df, period, 'close_change')
    rangema = zema(df, period, 'delta')

    upper = mid + rangema * mult
    lower = mid - rangema * mult

    return upper, rangema, lower


def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']


def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc


def linear_decay(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
    """
    Simple linear decay function. Decays from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (start - end) / (end_time - start_time)

    return max(end, start - (rate * time))


# #####################################################################################################

class ConsensusShort(IStrategy):
    """
    come from https://github.com/werkkrew/freqtrade-strategies/blob/main/strategies/archived/consensus_strat.py
    Author:werkkrew
    """
    minimal_roi = {
        "0": 100
    }

    stoploss = -0.99
    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count: int = 30

    can_short = True

    use_custom_stoploss = True

    custom_trade_info = {}

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    buy_optimize = True
    buy_score_short = IntParameter(low=0, high=100, default=20, space='buy', optimize=buy_optimize)

    leverage_optimize = True
    leverage_num = IntParameter(low=1, high=10, default=1, space='buy', optimize=leverage_optimize)

    protect_optimize = True
    cooldown_lookback = IntParameter(1, 240, default=5, space="protection", optimize=protect_optimize)
    max_drawdown_lookback = IntParameter(1, 288, default=12, space="protection", optimize=protect_optimize)
    max_drawdown_trade_limit = IntParameter(1, 20, default=5, space="protection", optimize=protect_optimize)
    max_drawdown_stop_duration = IntParameter(1, 288, default=12, space="protection", optimize=protect_optimize)
    max_allowed_drawdown = DecimalParameter(0.10, 0.50, default=0.20, decimals=2, space="protection",
                                            optimize=protect_optimize)
    stoploss_guard_lookback = IntParameter(1, 288, default=12, space="protection", optimize=protect_optimize)
    stoploss_guard_trade_limit = IntParameter(1, 20, default=3, space="protection", optimize=protect_optimize)
    stoploss_guard_stop_duration = IntParameter(1, 288, default=12, space="protection", optimize=protect_optimize)

    # custom exit
    csell_pullback_amount = DecimalParameter(0.005, 0.15, default=0.01, space='sell', load=True, optimize=True)
    csell_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)
    csell_roi_start = DecimalParameter(0.01, 0.15, default=0.01, space='sell', load=True, optimize=True)
    csell_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    csell_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    csell_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    csell_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    csell_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    csell_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)

    # Custom Stoploss
    cstop_loss_threshold = DecimalParameter(-0.35, -0.01, default=-0.03, space='sell', load=True, optimize=True)
    cstop_bail_how = CategoricalParameter(['roc', 'time', 'any', 'none'], default='none', space='sell', load=True,
                                          optimize=True)
    cstop_bail_roc = DecimalParameter(-5.0, -1.0, default=-3.0, space='sell', load=True, optimize=True)
    cstop_bail_time = IntParameter(60, 1440, default=720, space='sell', load=True, optimize=True)
    cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)

    # Protection hyperspace params:
    # Protection hyperspace params:
    protection_params = {
        "cooldown_lookback": 5,
        "max_drawdown_lookback": 12,
        "max_drawdown_trade_limit": 5,
        "max_drawdown_stop_duration": 12,
        "max_allowed_drawdown": 0.2,
        "stoploss_guard_lookback": 12,
        "stoploss_guard_trade_limit": 3,
        "stoploss_guard_stop_duration": 12
    }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value,
                "trade_limit": self.max_drawdown_trade_limit.value,
                "stop_duration_candles": self.max_drawdown_stop_duration.value,
                "max_allowed_drawdown": self.max_allowed_drawdown.value
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value,
                "trade_limit": self.stoploss_guard_trade_limit.value,
                "stop_duration_candles": self.stoploss_guard_stop_duration.value,
                "only_per_pair": False
            }
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Consensus strategy
        # add c.evaluate_indicator bellow to include it in the consensus score (look at
        # consensus.py in freqtrade technical)
        # add custom indicator with c.evaluate_consensus(prefix=<indicator name>)
        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}
            if 'had-trend' not in self.custom_trade_info[metadata["pair"]]:
                self.custom_trade_info[metadata['pair']]['had-trend'] = False

        c = Consensus(dataframe)
        c.evaluate_rsi()
        c.evaluate_stoch()
        c.evaluate_macd_cross_over()
        c.evaluate_macd()
        c.evaluate_hull()
        c.evaluate_vwma()
        c.evaluate_tema(period=12)
        c.evaluate_ema(period=24)
        c.evaluate_sma(period=12)
        c.evaluate_laguerre()
        c.evaluate_osc()
        c.evaluate_cmf()
        c.evaluate_cci()
        c.evaluate_cmo()
        c.evaluate_ichimoku()
        c.evaluate_ultimate_oscilator()
        c.evaluate_williams()
        c.evaluate_momentum()
        c.evaluate_adx()
        dataframe['consensus_buy'] = c.score()['buy']
        dataframe['consensus_sell'] = c.score()['sell']

        dataframe['rmi'] = RMI(dataframe, length=24, mom=5)
        dataframe['rmi-down'] = np.where(dataframe['rmi'] < dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-down-trend'] = np.where(dataframe['rmi-down'].rolling(5).sum() >= 3, 1, 0)

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup < ssldown, 'down', 'up')

        # Trends, Peaks and Crosses
        dataframe['candle-down'] = np.where(dataframe['close'] < dataframe['open'], 1, 0)
        dataframe['candle-down-trend'] = np.where(dataframe['candle-down'].rolling(5).sum() >= 3, 1, 0)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['consensus_buy'] < self.buy_score_short.value)
            ),
            'enter_short'] = 1

        dataframe.loc[
            (
            ),
            'enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(), 'exit_short'] = 0

        dataframe.loc[(), 'exit_long'] = 0

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value

    """
    Custom Stoploss
    """

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        in_trend = self.custom_trade_info[trade.pair]['had-trend']

        # Determine how we sell when we are in a loss
        if current_profit < self.cstop_loss_threshold.value:
            if self.cstop_bail_how.value == 'roc' or self.cstop_bail_how.value == 'any':
                # Dynamic bailout based on rate of change
                if last_candle['sroc'] <= self.cstop_bail_roc.value:
                    return 0.01
            if self.cstop_bail_how.value == 'time' or self.cstop_bail_how.value == 'any':
                # Dynamic bailout based on time, unless time_trend is true and there is a potential reversal
                if trade_dur > self.cstop_bail_time.value:
                    if self.cstop_bail_time_trend.value and in_trend:
                        return 1
                    else:
                        return 0.01
        return 1

    """
    Custom Sell
    """

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.csell_pullback_amount.value))
        in_trend = False

        # Determine our current ROI point based on the defined type
        if self.csell_roi_type.value == 'static':
            min_roi = self.csell_roi_start.value
        elif self.csell_roi_type.value == 'decay':
            min_roi = linear_decay(self.csell_roi_start.value, self.csell_roi_end.value, 0, self.csell_roi_time.value,
                                   trade_dur)
        elif self.csell_roi_type.value == 'step':
            if trade_dur < self.csell_roi_time.value:
                min_roi = self.csell_roi_start.value
            else:
                min_roi = self.csell_roi_end.value

        # Determine if there is a trend
        if self.csell_trend_type.value == 'rmi' or self.csell_trend_type.value == 'any':
            if last_candle['rmi-down-trend'] == 1:
                in_trend = True
        if self.csell_trend_type.value == 'ssl' or self.csell_trend_type.value == 'any':
            if last_candle['ssl-dir'] == 'down':
                in_trend = True
        if self.csell_trend_type.value == 'candle' or self.csell_trend_type.value == 'any':
            if last_candle['candle-down-trend'] == 1:
                in_trend = True

        # Don't sell if we are in a trend unless the pullback threshold is met
        if in_trend and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful sell message later
            self.custom_trade_info[trade.pair]['had-trend'] = True
            # If pullback is enabled and profit has pulled back allow a sell, maybe
            if self.csell_pullback.value and (current_profit <= pullback_value):
                if self.csell_pullback_respect_roi.value and current_profit > min_roi:
                    return 'intrend_pullback_roi'
                elif not self.csell_pullback_respect_roi.value:
                    if current_profit > min_roi:
                        return 'intrend_pullback_roi'
                    else:
                        return 'intrend_pullback_noroi'
            # We are in a trend and pullback is disabled or has not happened or various criteria were not met, hold
            return None
        # If we are not in a trend, just use the roi value
        elif not in_trend:
            if self.custom_trade_info[trade.pair]['had-trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_roi'
                elif not self.csell_endtrend_respect_roi.value:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_noroi'
            elif current_profit > min_roi:
                return 'notrend_roi'
        else:
            return None
