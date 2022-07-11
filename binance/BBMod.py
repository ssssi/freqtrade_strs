from typing import Dict, List

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas_ta as pta
import pandas as pd

from skopt.space import Dimension, Integer

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime
from freqtrade.strategy import DecimalParameter, IntParameter, informative, stoploss_from_open, CategoricalParameter
from functools import reduce
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# custom indicators
# ##################################################################################################
def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


def ewo(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


def vwap_b(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def top_percent_change(dataframe: DataFrame, length: int) -> float:
    """
    Percentage change of the current close from the range maximum Open price

    :param dataframe: DataFrame The original OHLC dataframe
    :param length: int The length to look back
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']


# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    wr = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
    )

    return wr * -100


def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']


def rmi(dataframe, *, length=20, mom=5):

    df = dataframe.copy()
    df["maxup"] = (df["close"] - df["close"].shift(mom)).clip(lower=0)
    df["maxdown"] = (df["close"].shift(mom) - df["close"]).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price="maxup", timeperiod=length)
    df["emaDec"] = ta.EMA(df, price="maxdown", timeperiod=length)

    df["RMI"] = np.where(df["emaDec"] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))
    return df["RMI"]

# #####################################################################################################


class BBMod(IStrategy):

    minimal_roi = {
        "0": 100
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True
    startup_candle_count = 200

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
    stoploss = -0.99

    # Custom stoploss
    use_custom_stoploss = True

    # Buy params
    leverage_optimize = False
    leverage_num = IntParameter(low=1, high=3, default=3, space='buy', optimize=leverage_optimize)

    buy_con_op = True
    buy_is_bb_checked_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_sqzmom_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_ewo_2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_r_deadfish_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_clucHA_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_cofi_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_gumbo_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_local_uptrend_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_local_uptrend2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_local_dip_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_ewo_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_nfi_32_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_nfix_39_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)
    buy_is_vwap_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_con_op)

    is_optimize_dip = True
    buy_rmi = IntParameter(30, 50, default=35, space='buy', optimize=is_optimize_dip)
    buy_cci = IntParameter(-135, -90, default=-133, space='buy', optimize=is_optimize_dip)
    buy_srsi_fk = IntParameter(30, 50, default=25, space='buy', optimize=is_optimize_dip)
    buy_cci_length = IntParameter(25, 45, default=25, space='buy', optimize=is_optimize_dip)
    buy_rmi_length = IntParameter(8, 20, default=8, space='buy', optimize=is_optimize_dip)

    is_optimize_break = True
    buy_bb_width = DecimalParameter(0.065, 0.135, default=0.095, space='buy', optimize=is_optimize_break)
    buy_bb_delta = DecimalParameter(0.018, 0.035, default=0.025, space='buy', optimize=is_optimize_break)
    break_closedelta = DecimalParameter(12.0, 18.0, default=15.0, space='buy', optimize=is_optimize_break)
    break_buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, space='buy', optimize=is_optimize_break)

    is_optimize_local_uptrend = True
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.025, space='buy', optimize=is_optimize_local_uptrend)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, space='buy', optimize=is_optimize_local_uptrend)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, space='buy', optimize=is_optimize_local_uptrend)

    is_optimize_local_uptrend2 = True
    buy_bb_factor2 = DecimalParameter(0.990, 0.999, default=0.995, space='buy', optimize=is_optimize_local_uptrend2)

    is_optimize_local_dip = True
    buy_ema_diff_local_dip = DecimalParameter(0.022, 0.027, default=0.025, space='buy', optimize=is_optimize_local_dip)
    buy_ema_high_local_dip = DecimalParameter(0.90, 1.2, default=0.942, space='buy', optimize=is_optimize_local_dip)
    buy_closedelta_local_dip = DecimalParameter(12.0, 18.0, default=15.0, space='buy', optimize=is_optimize_local_dip)
    buy_rsi_local_dip = IntParameter(15, 45, default=28, space='buy', optimize=is_optimize_local_dip)
    buy_crsi_local_dip = IntParameter(10, 18, default=10, space='buy', optimize=is_optimize_local_dip)

    is_optimize_ewo = True
    buy_rsi_fast = IntParameter(35, 50, default=45, space='buy', optimize=is_optimize_ewo)
    buy_rsi = IntParameter(15, 35, default=35, space='buy', optimize=is_optimize_ewo)
    buy_ewo = DecimalParameter(-6.0, 5, default=-5.585, space='buy', optimize=is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.942, space='buy', optimize=is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.084, space='buy', optimize=is_optimize_ewo)

    is_optimize_nfix_39 = True
    buy_nfix_39_ema = DecimalParameter(0.9, 1.2, default=0.97, space='buy', optimize=is_optimize_nfix_39)

    is_optimize_sqzmom_protection = True
    buy_sqzmom_ema = DecimalParameter(0.9, 1.2, default=0.97, space='buy', optimize=is_optimize_sqzmom_protection)
    buy_sqzmom_ewo = DecimalParameter(-12, 12, default=0, space='buy', optimize=is_optimize_sqzmom_protection)
    buy_sqzmom_r14 = DecimalParameter(-100, -22, default=-50, space='buy', optimize=is_optimize_sqzmom_protection)

    is_optimize_ewo_2 = True
    buy_rsi_fast_ewo_2 = IntParameter(15, 50, default=45, space='buy', optimize=is_optimize_ewo_2)
    buy_rsi_ewo_2 = IntParameter(15, 50, default=35, space='buy', optimize=is_optimize_ewo_2)
    buy_ema_low_2 = DecimalParameter(0.90, 1.2, default=0.970, space='buy', optimize=is_optimize_ewo_2)
    buy_ema_high_2 = DecimalParameter(0.90, 1.2, default=1.087, space='buy', optimize=is_optimize_ewo_2)
    buy_ewo_high_2 = DecimalParameter(2, 12, default=4.179, space='buy', optimize=is_optimize_ewo_2)

    is_optimize_r_deadfish = True
    buy_r_deadfish_ema = DecimalParameter(0.90, 1.2, default=1.087, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_bb_factor = DecimalParameter(0.90, 1.2, default=1.0, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_cti = DecimalParameter(-0.6, -0.0, default=-0.5, space='buy', optimize=is_optimize_r_deadfish)
    buy_r_deadfish_r14 = DecimalParameter(-60, -44, default=-60, space='buy', optimize=is_optimize_r_deadfish)

    is_optimize_cofi = True
    buy_ema_cofi = DecimalParameter(0.94, 1.2, default=0.97, space='buy', optimize=is_optimize_cofi)
    buy_fastk = IntParameter(0, 40, default=20, space='buy', optimize=is_optimize_cofi)
    buy_fastd = IntParameter(0, 40, default=20, space='buy', optimize=is_optimize_cofi)
    buy_adx = IntParameter(0, 30, default=30, space='buy', optimize=is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, space='buy', optimize=is_optimize_cofi)
    buy_cofi_cti = DecimalParameter(-0.9, -0.0, default=-0.5, space='buy', optimize=is_optimize_cofi)
    buy_cofi_r14 = DecimalParameter(-100, -44, default=-60, space='buy', optimize=is_optimize_cofi)

    is_optimize_clucha = True
    buy_clucha_bbdelta_close = DecimalParameter(0.01, 0.05, default=0.02206, space='buy', optimize=is_optimize_clucha)
    buy_clucha_bbdelta_tail = DecimalParameter(0.7, 1.2, default=1.02515, space='buy', optimize=is_optimize_clucha)
    buy_clucha_closedelta_close = DecimalParameter(0.001, 0.05, default=0.04401, space='buy',
                                                   optimize=is_optimize_clucha)
    buy_clucha_rocr_1h = DecimalParameter(0.1, 1.0, default=0.47782, space='buy', optimize=is_optimize_clucha)

    is_optimize_gumbo = True
    buy_gumbo_ema = DecimalParameter(0.9, 1.2, default=0.97, space='buy', optimize=is_optimize_gumbo)
    buy_gumbo_ewo_low = DecimalParameter(-12.0, 5, default=-5.585, space='buy', optimize=is_optimize_gumbo)
    buy_gumbo_cti = DecimalParameter(-0.9, -0.0, default=-0.5, space='buy', optimize=is_optimize_gumbo)
    buy_gumbo_r14 = DecimalParameter(-100, -44, default=-60, space='buy', optimize=is_optimize_gumbo)

    # custom stoploss
    trailing_optimize = True
    pHSL = DecimalParameter(-0.990, -0.040, default=-0.1, decimals=3, space='sell', optimize=False)
    pPF_1 = DecimalParameter(0.008, 0.100, default=0.03, decimals=3, space='sell', optimize=False)
    pSL_1 = DecimalParameter(0.02, 0.030, default=0.025, decimals=3, space='sell', optimize=trailing_optimize)
    pPF_2 = DecimalParameter(0.040, 0.200, default=0.080, decimals=3, space='sell', optimize=False)
    pSL_2 = DecimalParameter(0.070, 0.080, default=0.075, decimals=3, space='sell', optimize=trailing_optimize)

    ############################################################################
    class HyperOpt:
        @staticmethod
        def generate_roi_table(params: Dict) -> Dict[int, float]:
            roi_table = {params['roi_t1']: 0}
            return roi_table

        @staticmethod
        def roi_space() -> List[Dimension]:
            roi_min_time = 10
            roi_max_time = 360

            roi_limits = {
                'roi_t1_min': int(roi_min_time),
                'roi_t1_max': int(roi_max_time)
            }

            return [
                Integer(roi_limits['roi_t1_min'], roi_limits['roi_t1_max'], name='roi_t1')
            ]

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if self.can_short:
            if (-1 + ((1 - sl_profit) / (1 - current_profit))) <= 0:
                return 1
        else:
            if (1 - ((1 + sl_profit) / (1 + current_profit))) <= 0:
                return 1

        return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_close'] = heikinashi['close']
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        # # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['T3'] = T3(dataframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        # Other BB checks
        dataframe['bb_width'] = (
                (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])
        dataframe['bb_delta'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])

        # BinH
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        # SMA
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_28'] = ta.SMA(dataframe, timeperiod=28)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # CRSI (3, 2, 100)
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi'] = (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(
            dataframe['close'], 100)) / 3

        # EMA
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # Elliot
        dataframe['EWO'] = ewo(dataframe, 50, 200)

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # BB 40
        bollinger2_40 = qtpylib.bollinger_bands(ha_typical_price(dataframe), window=40, stds=2)
        dataframe['bb_lowerband2_40'] = bollinger2_40['lower']
        dataframe['bb_middleband2_40'] = bollinger2_40['mid']
        dataframe['bb_upperband2_40'] = bollinger2_40['upper']

        # ClucHA
        dataframe['bb_delta_cluc'] = (dataframe['bb_middleband2_40'] - dataframe['bb_lowerband2_40']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        # vmap indicators
        vwap_low, vwap, vwap_high = vwap_b(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        dataframe['tcp_percent_4'] = top_percent_change(dataframe, 4)

        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = rmi(dataframe, length=val, mom=4)

        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']

        # True range
        dataframe['trange'] = ta.TRANGE(dataframe)

        # KC
        dataframe['range_ma_28'] = ta.SMA(dataframe['trange'], 28)
        dataframe['kc_upperband_28_1'] = dataframe['sma_28'] + dataframe['range_ma_28']
        dataframe['kc_lowerband_28_1'] = dataframe['sma_28'] - dataframe['range_ma_28']

        # Linreg
        dataframe['hh_20'] = ta.MAX(dataframe['high'], 20)
        dataframe['ll_20'] = ta.MIN(dataframe['low'], 20)
        dataframe['avg_hh_ll_20'] = (dataframe['hh_20'] + dataframe['ll_20']) / 2
        dataframe['avg_close_20'] = ta.SMA(dataframe['close'], 20)
        dataframe['avg_val_20'] = (dataframe['avg_hh_ll_20'] + dataframe['avg_close_20']) / 2
        dataframe['linreg_val_20'] = ta.LINEARREG(dataframe['close'] - dataframe['avg_val_20'], 20, 0)

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        dataframe['r_14'] = williams_r(dataframe, period=14)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)

        # T3 Averag
        dataframe['T3'] = T3(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        is_bb_checked = (
                self.buy_is_bb_checked_enable.value &
                (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value) &
                (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk.value) &
                (dataframe['bb_delta'] > self.buy_bb_delta.value) &
                (dataframe['bb_width'] > self.buy_bb_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.break_closedelta.value / 1000) &  # from BinH
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.break_buy_bb_factor.value)
        )

        is_sqzmom = (
                self.buy_is_sqzmom_enable.value &
                (dataframe['bb_lowerband2'] < dataframe['kc_lowerband_28_1']) &
                (dataframe['bb_upperband2'] > dataframe['kc_upperband_28_1']) &
                (dataframe['linreg_val_20'].shift(2) > dataframe['linreg_val_20'].shift(1)) &
                (dataframe['linreg_val_20'].shift(1) < dataframe['linreg_val_20']) &
                (dataframe['linreg_val_20'] < 0) &
                (dataframe['close'] < dataframe['ema_13'] * self.buy_sqzmom_ema.value) &
                (dataframe['EWO'] < self.buy_sqzmom_ewo.value) &
                (dataframe['r_14'] < self.buy_sqzmom_r14.value)
        )

        is_ewo_2 = (
                self.buy_is_ewo_2_enable.value &
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_ewo_2.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low_2.value) &
                (dataframe['EWO'] > self.buy_ewo_high_2.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high_2.value) &
                (dataframe['rsi'] < self.buy_rsi_ewo_2.value)
        )

        is_r_deadfish = (
                self.buy_is_r_deadfish_enable.value &
                (dataframe['ema_100'] < dataframe['ema_200'] * self.buy_r_deadfish_ema.value) &
                (dataframe['bb_width'] > self.buy_r_deadfish_bb_width.value) &
                (dataframe['close'] < dataframe['bb_middleband2'] * self.buy_r_deadfish_bb_factor.value) &
                (dataframe['volume_mean_12'] > dataframe['volume_mean_24'] * self.buy_r_deadfish_volume_factor.value) &
                (dataframe['cti'] < self.buy_r_deadfish_cti.value) &
                (dataframe['r_14'] < self.buy_r_deadfish_r14.value)
        )

        is_clucha = (
                self.buy_is_clucHA_enable.value &
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h.value) &
                (
                        (dataframe['bb_lowerband2_40'].shift() > 0) &
                        (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close.value) &
                        (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close.value) &
                        (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail.value) &
                        (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                        (dataframe['ha_close'] < dataframe['ha_close'].shift())
                )
        )

        is_cofi = (
                self.buy_is_cofi_enable.value &
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value) &
                (dataframe['cti'] < self.buy_cofi_cti.value) &
                (dataframe['r_14'] < self.buy_cofi_r14.value)
        )

        is_gumbo = (
                self.buy_is_gumbo_enable.value &
                (dataframe['EWO'] < self.buy_gumbo_ewo_low.value) &
                (dataframe['bb_middleband2_1h'] >= dataframe['T3_1h']) &
                (dataframe['T3'] <= dataframe['ema_8'] * self.buy_gumbo_ema.value) &
                (dataframe['cti'] < self.buy_gumbo_cti.value) &
                (dataframe['r_14'] < self.buy_gumbo_r14.value)
        )

        is_local_uptrend = (  # from NFI next gen, credit goes to @iterativ
                self.buy_is_local_uptrend_enable.value &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000)
        )

        is_local_uptrend2 = (  # use origin bb_rpb_tsl value
                self.buy_is_local_uptrend2_enable.value &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * 0.026) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor2.value) &
                (dataframe['closedelta'] > dataframe['close'] * 17.922 / 1000)
        )

        is_local_dip = (
                self.buy_is_local_dip_enable.value &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff_local_dip.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_ema_high_local_dip.value) &
                (dataframe['rsi'] < self.buy_rsi_local_dip.value) &
                (dataframe['crsi'] > self.buy_crsi_local_dip.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta_local_dip.value / 1000)
        )

        is_ewo = (  # from SMA offset
                self.buy_is_ewo_enable.value &
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
                (dataframe['EWO'] > self.buy_ewo.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
                (dataframe['rsi'] < self.buy_rsi.value)
        )

        is_nfi_32 = (
                self.buy_is_nfi_32_enable.value &
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 46) &
                (dataframe['rsi'] > 19) &
                (dataframe['close'] < dataframe['sma_15'] * 0.942) &
                (dataframe['cti'] < -0.86)
        )

        is_nfix_39 = (
                self.buy_is_nfix_39_enable.value &
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['bb_lowerband2_40'].shift().gt(0)) &
                (dataframe['bb_delta_cluc'].gt(dataframe['close'] * 0.056)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.01)) &
                (dataframe['tail'].lt(dataframe['bb_delta_cluc'] * 0.5)) &
                (dataframe['close'].lt(dataframe['bb_lowerband2_40'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['close'] > dataframe['ema_13'] * self.buy_nfix_39_ema.value)
        )

        is_vwap = (
                self.buy_is_vwap_enable.value &
                (dataframe['close'] < dataframe['vwap_low']) &
                (dataframe['tcp_percent_4'] > 0.04) &
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi'] < 35) &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60) &
                (dataframe['volume'] > 0)
        )

        conditions.append(is_bb_checked)
        dataframe.loc[is_bb_checked, 'enter_tag'] += 'bb '

        conditions.append(is_sqzmom)
        dataframe.loc[is_sqzmom, 'enter_tag'] += 'sqzmom '

        conditions.append(is_ewo_2)
        dataframe.loc[is_ewo_2, 'enter_tag'] += 'ewo2 '

        conditions.append(is_r_deadfish)
        dataframe.loc[is_r_deadfish, 'enter_tag'] += 'r_deadfish '

        conditions.append(is_clucha)
        dataframe.loc[is_clucha, 'enter_tag'] += 'clucHA '

        conditions.append(is_cofi)
        dataframe.loc[is_cofi, 'enter_tag'] += 'cofi '

        conditions.append(is_gumbo)
        dataframe.loc[is_gumbo, 'enter_tag'] += 'gumbo '

        conditions.append(is_local_uptrend)
        dataframe.loc[is_local_uptrend, 'enter_tag'] += 'local_uptrend '

        conditions.append(is_local_dip)
        dataframe.loc[is_local_dip, 'enter_tag'] += 'local_dip '

        conditions.append(is_ewo)
        dataframe.loc[is_ewo, 'enter_tag'] += 'ewo '

        conditions.append(is_nfi_32)
        dataframe.loc[is_nfi_32, 'enter_tag'] += 'nfi_32 '

        conditions.append(is_nfix_39)
        dataframe.loc[is_nfix_39, 'enter_tag'] += 'nfix_39 '

        conditions.append(is_vwap)
        dataframe.loc[is_vwap, 'enter_tag'] += 'vwap '

        conditions.append(is_local_uptrend2)
        dataframe.loc[is_local_uptrend2, 'enter_tag'] += 'local_uptrend2 '

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value
