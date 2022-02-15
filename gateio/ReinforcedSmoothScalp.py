# --- Do not remove these libs ---
from functools import reduce
from typing import Optional

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, DecimalParameter, stoploss_from_open
from freqtrade.strategy import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, IntParameter
from pandas import DataFrame
from technical.util import resample_to_interval, resampled_merge
import numpy  # noqa
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime


class ReinforcedSmoothScalp(IStrategy):
    """
        this strategy is based around the idea of generating a lot of potentatils buys and make tiny profits on each trade

        we recommend to have at least 60 parallel trades at any time to cover non avoidable losses
    """

    sell_params = {
        # custom stoploss params, come from BB_RPB_TSL
        "pHSL": -0.99,
        "pPF_1": 0.02,
        "pPF_2": 0.05,
        "pSL_1": 0.02,
        "pSL_2": 0.04,
    }

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 100
    }
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    # should not be below 3% loss

    stoploss = -0.99
    # Optimal timeframe for the strategy
    # the shorter the better
    timeframe = '1m'

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Custom stoploss
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 50

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

    # resample factor to establish our general trend. Basically don't buy if a trend is not given
    resample_factor = 5

    buy_adx = IntParameter(20, 50, default=32, space='buy')
    buy_fastd = IntParameter(15, 45, default=30, space='buy')
    buy_fastk = IntParameter(15, 45, default=26, space='buy')
    buy_mfi = IntParameter(10, 25, default=22, space='buy')
    buy_adx_enabled = BooleanParameter(default=True, space='buy')
    buy_fastd_enabled = BooleanParameter(default=True, space='buy')
    buy_fastk_enabled = BooleanParameter(default=False, space='buy')
    buy_mfi_enabled = BooleanParameter(default=True, space='buy')

    sell_adx = IntParameter(50, 100, default=53, space='sell')
    sell_cci = IntParameter(100, 200, default=183, space='sell')
    sell_fastd = IntParameter(50, 100, default=79, space='sell')
    sell_fastk = IntParameter(50, 100, default=70, space='sell')
    sell_mfi = IntParameter(75, 100, default=92, space='sell')

    sell_adx_enabled = BooleanParameter(default=False, space='sell')
    sell_cci_enabled = BooleanParameter(default=True, space='sell')
    sell_fastd_enabled = BooleanParameter(default=True, space='sell')
    sell_fastk_enabled = BooleanParameter(default=True, space='sell')
    sell_mfi_enabled = BooleanParameter(default=False, space='sell')

    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
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

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tf_res = timeframe_to_minutes(self.timeframe) * 5
        df_res = resample_to_interval(dataframe, tf_res)
        df_res['sma'] = ta.SMA(df_res, 50, price='close')
        dataframe = resampled_merge(dataframe, df_res, fill_na=True)
        dataframe['resample_sma'] = dataframe[f'resample_{tf_res}_sma']

        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe)

        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] < self.buy_mfi.value)
        if self.buy_fastd_enabled.value:
            conditions.append(dataframe['fastd'] < self.buy_fastd.value)
        if self.buy_fastk_enabled.value:
            conditions.append(dataframe['fastk'] < self.buy_fastk.value)
        if self.buy_adx_enabled.value:
            conditions.append(dataframe['adx'] > self.buy_adx.value)

        # Some static conditions which always apply
        conditions.append(qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
        conditions.append(dataframe['resample_sma'] < dataframe['close'])

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        # Some static conditions which always apply
        conditions.append(dataframe['open'] > dataframe['ema_high'])

        if self.sell_mfi_enabled.value:
            conditions.append(dataframe['mfi'] > self.sell_mfi.value)
        if self.sell_fastd_enabled.value:
            conditions.append(dataframe['fastd'] > self.sell_fastd.value)
        if self.sell_fastk_enabled.value:
            conditions.append(dataframe['fastk'] > self.sell_fastk.value)
        if self.sell_adx_enabled.value:
            conditions.append(dataframe['adx'] < self.sell_adx.value)
        if self.sell_cci_enabled.value:
            conditions.append(dataframe['cci'] > self.sell_cci.value)

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe


class ReinforcedSmoothScalpDCA(ReinforcedSmoothScalp):
    position_adjustment_enable = True

    max_rebuy_orders = 1
    max_rebuy_multiplier = 2

    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: Optional[str], **kwargs) -> float:

        if (self.config['position_adjustment_enable'] is True) and (self.config['stake_amount'] == 'unlimited'):
            return proposed_stake / self.max_rebuy_multiplier
        else:
            return proposed_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        if (self.config['position_adjustment_enable'] is False) or (current_profit > -0.08):
            return None

        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        # Maximum 2 rebuys, equal stake as the original
        if 0 < count_of_buys <= self.max_rebuy_orders:
            try:
                # This returns first order stake size
                stake_amount = filled_buys[0].cost
                # This then calculates current safety order size
                stake_amount = stake_amount
                return stake_amount
            except Exception as exception:
                return None

        return None