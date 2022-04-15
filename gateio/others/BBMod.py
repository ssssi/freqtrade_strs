import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas_ta as pta

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter, stoploss_from_open
from functools import reduce

TMP_HOLD = []


# --------------------------------
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


class BBMod(IStrategy):
    """
        BBMod1 modified from BB_RPB_TSL ( https://github.com/jilv220/BB_RPB_TSL )
        @author jilv220
    """

    # buy space
    buy_params = {
        ##
        "buy_bb_factor": 0.999,
        "buy_closedelta": 12.148,
        "buy_ema_diff": 0.022,
        ##
        "buy_ema_high": 0.968,
        "buy_ema_low": 0.935,
        "buy_ewo": -5.001,
        "buy_rsi": 23,
        "buy_rsi_fast": 44,
        ##
        "buy_closedelta_local_dip": 12.044,
        "buy_ema_diff_local_dip": 0.024,
        "buy_ema_high_local_dip": 1.014,
        "buy_rsi_local_dip": 21,
        "buy_crsi_local_dip": 10,
        ##
        "buy_nfix_39_ema": 0.912
    }

    # sell space
    sell_params = {
        "high_offset_2": 0.997,
        "pHSL": -0.18,
        "pPF_1": 0.019,
        "pPF_2": 0.05,
        "pSL_1": 0.017,
        "pSL_2": 0.045,
    }

    minimal_roi = {
        "0": 100
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True
    startup_candle_count = 120

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'emergency_exit': 'limit',
        'force_entry': 'limit',
        'force_exit': "limit",
        'stoploss': 'limit',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # Disabled
    stoploss = -0.99

    # Custom stoploss
    use_custom_stoploss = True

    # Buy params
    is_optimize_local_uptrend = True
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.025, optimize=is_optimize_local_uptrend)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, optimize=is_optimize_local_uptrend)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, optimize=is_optimize_local_uptrend)

    is_optimize_local_dip = True
    buy_ema_diff_local_dip = DecimalParameter(0.022, 0.027, default=0.025, optimize=is_optimize_local_dip)
    buy_ema_high_local_dip = DecimalParameter(0.90, 1.2, default=0.942, optimize=is_optimize_local_dip)
    buy_closedelta_local_dip = DecimalParameter(12.0, 18.0, default=15.0, optimize=is_optimize_local_dip)
    buy_rsi_local_dip = IntParameter(15, 45, default=28, optimize=is_optimize_local_dip)
    buy_crsi_local_dip = IntParameter(10, 18, default=10, optimize=is_optimize_local_dip)

    is_optimize_ewo = True
    buy_rsi_fast = IntParameter(35, 50, default=45, optimize=is_optimize_ewo)
    buy_rsi = IntParameter(15, 35, default=35, optimize=is_optimize_ewo)
    buy_ewo = DecimalParameter(-6.0, 5, default=-5.585, optimize=is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.942, optimize=is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.084, optimize=is_optimize_ewo)

    is_optimize_nfix_39 = True
    buy_nfix_39_ema = DecimalParameter(0.9, 1.2, default=0.97, optimize=is_optimize_nfix_39)

    # rng sell
    high_offset_2 = DecimalParameter(0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=False)

    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    ############################################################################

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        # EMA
        informative_1h['ema_8'] = ta.EMA(informative_1h, timeperiod=8)
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        # CRSI (3, 2, 100)
        crsi_closechange = informative_1h['close'] / informative_1h['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        informative_1h['crsi'] = (ta.RSI(informative_1h['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) +
                                  ta.ROC(informative_1h['close'], 100)) / 3

        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband2'] = bollinger2['lower']
        informative_1h['bb_middleband2'] = bollinger2['mid']
        informative_1h['bb_upperband2'] = bollinger2['upper']
        informative_1h['bb_width'] = ((informative_1h['bb_upperband2'] - informative_1h['bb_lowerband2']) /
                                      informative_1h['bb_middleband2'])

        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # Elliot
        informative_1h['EWO'] = ewo(informative_1h, 50, 200)

        return informative_1h

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        stoploss = self.pHSL.value
        pf_1 = self.pPF_1.value
        sl_1 = self.pSL_1.value
        pf_2 = self.pPF_2.value
        sl_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit >= pf_2:
            sl_profit = sl_2 + (current_profit - pf_2)
        elif current_profit >= pf_1:
            sl_profit = sl_1 + ((current_profit - pf_1) * (sl_2 - sl_1) / (pf_2 - pf_1))
        else:
            sl_profit = stoploss

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle_1 = dataframe.iloc[-2].squeeze()

        if current_profit >= 0.019:
            return None

        if (
                (current_profit < 0.019)
                and (last_candle['close'] > last_candle['sma_9'])
                and (last_candle['close'] > last_candle['ema_24'] * self.high_offset_2.value)
                and (last_candle['rsi'] > 50)
                and (last_candle['rsi_fast'] > last_candle['rsi_slow'])
        ):
            if (
                    (last_candle['close'] > last_candle['bb_middleband2'])
            ):
                if trade.id not in TMP_HOLD:
                    TMP_HOLD.append(trade.id)
                    return None

        for i in TMP_HOLD:
            if trade.id == i and (last_candle['close'] < previous_candle_1['close']):
                TMP_HOLD.remove(i)
                return "sell_drop"

    @staticmethod
    def normal_tf_indicators(dataframe: DataFrame) -> DataFrame:

        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        # BinH
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        # SMA
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_21'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['sma_28'] = ta.SMA(dataframe, timeperiod=28)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma_75'] = ta.SMA(dataframe, timeperiod=75)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # CRSI (3, 2, 100)
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi'] = (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(
            dataframe['close'], 100)) / 3

        # EMA
        dataframe['ema_4'] = ta.EMA(dataframe, timeperiod=4)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_24'] = ta.EMA(dataframe, timeperiod=24)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_49'] = ta.EMA(dataframe, timeperiod=49)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
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
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        # vmap indicators
        vwap_low, vwap, vwap_high = vwap_b(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        dataframe['tcp_percent_4'] = top_percent_change(dataframe, 4)

        return dataframe

    ############################################################################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        is_local_uptrend = (  # from NFI next gen, credit goes to @iterativ
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000)
        )

        is_local_uptrend2 = (  # use origin bb_rpb_tsl value
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * 0.026) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * 17.922 / 1000)
        )

        is_local_dip = (
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff_local_dip.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_ema_high_local_dip.value) &
                (dataframe['rsi'] < self.buy_rsi_local_dip.value) &
                (dataframe['crsi'] > self.buy_crsi_local_dip.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta_local_dip.value / 1000)
        )

        is_ewo = (  # from SMA offset
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
                (dataframe['EWO'] > self.buy_ewo.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
                (dataframe['rsi'] < self.buy_rsi.value)
        )

        is_nfi_32 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 46) &
                (dataframe['rsi'] > 19) &
                (dataframe['close'] < dataframe['sma_15'] * 0.942) &
                (dataframe['cti'] < -0.86)
        )

        is_nfix_39 = (
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
                (dataframe['close'] < dataframe['vwap_low']) &
                (dataframe['tcp_percent_4'] > 0.04) &
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi'] < 35) &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60) &
                (dataframe['volume'] > 0)
        )

        conditions.append(is_local_uptrend)  # ~3.28 / 92.4% / 69.72%
        dataframe.loc[is_local_uptrend, 'enter_tag'] += 'local_uptrend '

        conditions.append(is_local_dip)  # ~0.76 / 91.1% / 15.54%
        dataframe.loc[is_local_dip, 'enter_tag'] += 'local_dip '

        conditions.append(is_ewo)  # ~0.92 / 92.0% / 43.74%      D
        dataframe.loc[is_ewo, 'enter_tag'] += 'ewo '

        conditions.append(is_nfi_32)  # ~0.78 / 92.0 % / 37.41%     D
        dataframe.loc[is_nfi_32, 'enter_tag'] += 'nfi_32 '

        conditions.append(is_nfix_39)  # ~5.33 / 91.8% / 58.57%      D
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

        return 1.0