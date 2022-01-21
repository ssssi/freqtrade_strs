# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas_ta as pta

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, DatetimeIndex, merge
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from functools import reduce
from technical.indicators import RMI, zema

# --------------------------------
def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity limit is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100


####################################################
# VWAP bands, come from vwap strategy

def vmap_b(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
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

####################################################

class BB_RPB_TSL_RNG_VWAP(IStrategy):
    '''
        BB_RPB_TSL
        @author jilv220
        Simple bollinger brand strategy inspired by this blog  ( https://hacks-for-life.blogspot.com/2020/12/freqtrade-notes.html )
        RPB, which stands for Real Pull Back, taken from ( https://github.com/GeorgeMurAlkh/freqtrade-stuff/blob/main/user_data/strategies/TheRealPullbackV2.py )
        The trailing custom stoploss taken from BigZ04_TSL from Perkmeister ( modded by ilya )
        I modified it to better suit my taste and added Hyperopt for this strategy.
    '''

    ##########################################################################

    # Hyperopt result area

    # buy space
    buy_params = {
        ##
        "buy_btc_safe": -289,
        "buy_btc_safe_1d": -0.05,
        ##
        "buy_threshold": 0.003,
        "buy_bb_factor": 0.999,
        "buy_bb_delta": 0.025,
        "buy_bb_width": 0.095,
        ##
        "buy_cci": -116,
        "buy_cci_length": 25,
        "buy_rmi": 49,
        "buy_rmi_length": 17,
        "buy_srsi_fk": 32,
        ##
        "buy_closedelta": 12.148,
        "buy_ema_diff": 0.022,
        ##
        "buy_adx": 20,
        "buy_fastd": 20,
        "buy_fastk": 22,
        "buy_ema_cofi": 0.98,
        "buy_ewo_high": 4.179,
        ##
        "buy_ema_high_2": 1.087,
        "buy_ema_low_2": 0.970,
        ##
    }

    # sell space
    sell_params = {
        "pHSL": -0.25,
        "pPF_1": 0.019,
        "pPF_2": 0.065,
        "pSL_1": 0.019,
        "pSL_2": 0.062,
        "sell_btc_safe": -389,
        "base_nb_candles_sell": 24,
        "high_offset": 0.991,
        "high_offset_2": 0.997        
    }

    # really hard to use this
    minimal_roi = {
        "0": 100,
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    # Disabled
    stoploss = -0.99

    # Custom stoploss
    use_custom_stoploss = True
    use_sell_signal = True

    process_only_new_candles = True
    startup_candle_count = 120

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'emergencysell': 'limit',
        'forcebuy': "limit",
        'forcesell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    ############################################################################

    ## Buy params

    is_optimize_dip = False
    buy_rmi = IntParameter(30, 50, default=35, optimize= is_optimize_dip)
    buy_cci = IntParameter(-135, -90, default=-133, optimize= is_optimize_dip)
    buy_srsi_fk = IntParameter(30, 50, default=25, optimize= is_optimize_dip)
    buy_cci_length = IntParameter(25, 45, default=25, optimize = is_optimize_dip)
    buy_rmi_length = IntParameter(8, 20, default=8, optimize = is_optimize_dip)

    is_optimize_break = False
    buy_bb_width = DecimalParameter(0.05, 0.2, default=0.15, optimize = is_optimize_break)
    buy_bb_delta = DecimalParameter(0.025, 0.08, default=0.04, optimize = is_optimize_break)

    is_optimize_local_dip = False
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_dip)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, optimize = False)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_dip)

    is_optimize_ewo = False
    buy_rsi_fast = IntParameter(35, 50, default=45, optimize = False)
    buy_rsi = IntParameter(15, 30, default=35, optimize = False)
    buy_ewo = DecimalParameter(-6.0, 5, default=-5.585, optimize = is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.942 , optimize = is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.084 , optimize = is_optimize_ewo)

    is_optimize_ewo_2 = False
    buy_ema_low_2 = DecimalParameter(0.96, 0.978, default=0.96 , optimize = is_optimize_ewo_2)
    buy_ema_high_2 = DecimalParameter(1.05, 1.2, default=1.09 , optimize = is_optimize_ewo_2)

    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    is_optimize_btc_safe = False
    buy_btc_safe = IntParameter(-300, 50, default=-200, optimize = is_optimize_btc_safe)
    buy_btc_safe_1d = DecimalParameter(-0.075, -0.025, default=-0.05, optimize = is_optimize_btc_safe)
    buy_threshold = DecimalParameter(0.003, 0.012, default=0.008, optimize = is_optimize_btc_safe)

    # Buy params toggle
    buy_is_dip_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_is_break_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)

    ## Sell params
    sell_btc_safe = IntParameter(-400, -300, default=-365, optimize = True)
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    high_offset          = DecimalParameter(0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2        = DecimalParameter(0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)      

    ## Trailing params

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

        informative_pairs = [("BTC/USDT", "5m")]

        return informative_pairs

    ############################################################################

    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        buy_tag = ''
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        buy_tags = buy_tag.split()

        if len(buy_tags) == 1 and "vwap" in buy_tags:
            PF_1 = 0.01
            SL_1 = 0.01
            PF_2 = 0.05
            SL_2 = 0.042

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    ############################################################################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        assert self.dp, "DataProvider is required for multiple timeframes."

        # Bollinger bands (hyperopt hard to implement)
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        # vmap indicators
        vwap_low, vwap, vwap_high = vmap_b(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        dataframe['tcp_percent_4'] = top_percent_change(dataframe, 4)


        ### BTC protection

        # BTC info
        inf_tf = '5m'
        informative = self.dp.get_pair_dataframe('BTC/USDT', timeframe=inf_tf)
        informative_past = informative.copy().shift(1)                                                                                                   # Get recent BTC info

        # BTC 5m dump protection
        informative_past_source = (informative_past['open'] + informative_past['close'] + informative_past['high'] + informative_past['low']) / 4        # Get BTC price
        informative_threshold = informative_past_source * self.buy_threshold.value                                                                       # BTC dump n% in 5 min
        informative_past_delta = informative_past['close'].shift(1) - informative_past['close']                                                          # should be positive if dump
        informative_diff = informative_threshold - informative_past_delta                                                                                # Need be larger than 0
        dataframe['btc_threshold'] = informative_threshold
        dataframe['btc_diff'] = informative_diff

        # BTC 1d dump protection
        informative_past_1d = informative.copy().shift(288)
        informative_past_source_1d = (informative_past_1d['open'] + informative_past_1d['close'] + informative_past_1d['high'] + informative_past_1d['low']) / 4
        dataframe['btc_5m'] = informative_past_source
        dataframe['btc_1d'] = informative_past_source_1d

        ### Other checks

        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])
        dataframe['bb_delta'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])
        dataframe['bb_bottom_cross'] = qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband3']).astype('int')

        # CCI hyperopt
        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cci'] = ta.CCI(dataframe, 26)
        dataframe['cci_long'] = ta.CCI(dataframe, 170)

        # RMI hyperopt
        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)
        #dataframe['rmi'] = RMI(dataframe, length=8, mom=4)

        # SRSI hyperopt ?
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        # BinH
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        # SMA
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # EMA
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)   
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)        

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, 50, 200)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)

        # Volume
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)


        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)        

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        if self.buy_is_dip_enabled.value:

            is_dip = (
                (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value) &
                (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk.value)
            )

            #conditions.append(is_dip)

        if self.buy_is_break_enabled.value:

            is_break = (

                (   (dataframe['bb_delta'] > self.buy_bb_delta.value)                                   #"buy_bb_delta": 0.025 0.036
                    &                                                                                   #"buy_bb_width": 0.095 0.133
                    (dataframe['bb_width'] > self.buy_bb_width.value)
                )
                &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) &    # from BinH
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value)
            )
            #conditions.append(is_break)

        is_local_uptrend = (                                                                            # from NFI next gen

                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 )
            )

        is_ewo = (                                                                                      # from SMA offset
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
                (dataframe['EWO'] > self.buy_ewo.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
                (dataframe['rsi'] < self.buy_rsi.value)
            )

        # disabled ewo2 and cofi buy condition
        # is_ewo_2 = (
        #         (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
        #         (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low_2.value) &
        #         (dataframe['EWO'] > self.buy_ewo_high.value) &
        #         (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high_2.value) &
        #         (dataframe['rsi'] < self.buy_rsi.value)
        #     )

        # is_cofi = (
        #         (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
        #         (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
        #         (dataframe['fastk'] < self.buy_fastk.value) &
        #         (dataframe['fastd'] < self.buy_fastd.value) &
        #         (dataframe['adx'] > self.buy_adx.value) &
        #         (dataframe['EWO'] > self.buy_ewo_high.value)
        #     )

        # NFI quick mode

        is_nfi_32 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 46) &
                (dataframe['rsi'] > 19) &
                (dataframe['close'] < dataframe['sma_15'] * 0.942) &
                (dataframe['cti'] < -0.86)
            )

        is_nfi_33 = (
                (dataframe['close'] < (dataframe['ema_13'] * 0.978)) &
                (dataframe['EWO'] > 8) &
                (dataframe['cti'] < -0.88) &
                (dataframe['rsi'] < 32) &
                (dataframe['r_14'] < -98.0) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 2.5))
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


        is_BB_checked = is_dip & is_break

        ## condition append
        conditions.append(is_BB_checked)          # ~1.7 89%
        dataframe.loc[is_BB_checked, 'buy_tag'] += 'bb '

        conditions.append(is_local_uptrend)       # ~3.84 90.2%
        dataframe.loc[is_local_uptrend, 'buy_tag'] += 'local uptrend '

        conditions.append(is_ewo)                 # ~2.26 93.5%
        dataframe.loc[is_ewo, 'buy_tag'] += 'ewo '

        # conditions.append(is_ewo_2)               # ~3.68 90.3%
        # dataframe.loc[is_ewo_2, 'buy_tag'] += 'ewo2 '
        #
        # conditions.append(is_cofi)                # ~3.21 90.8%
        # dataframe.loc[is_cofi, 'buy_tag'] += 'cofi '

        conditions.append(is_nfi_32)              # ~2.43 91.3%
        dataframe.loc[is_nfi_32, 'buy_tag'] += 'nfi 32 '

        conditions.append(is_nfi_33)              # ~0.11 100%
        dataframe.loc[is_nfi_33, 'buy_tag'] += 'nfi 33 '

        conditions.append(is_vwap)
        dataframe.loc[is_vwap, 'buy_tag'] += 'vwap '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'buy' ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (                   
                (dataframe['close'] > dataframe['sma_9'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi']>50)&
                (dataframe['volume'] > 0)&                
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])

            )
            |
            (
                (dataframe['sma_9'] > (dataframe['sma_9'].shift(1) + dataframe['sma_9'].shift(1)*0.005 )) &
                (dataframe['close'] < dataframe['hma_50'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast']>dataframe['rsi_slow'])
            )    
            
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
