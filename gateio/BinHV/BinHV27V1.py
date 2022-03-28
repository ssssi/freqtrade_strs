from datetime import datetime

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, stoploss_from_open, CategoricalParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import numpy  # noqa


class BinHV27V1(IStrategy):
    """

        strategy sponsored by user BinH from slack

    """

    minimal_roi = {
        "0": 1
    }

    sell_params = {
        # custom stoploss params, come from BB_RPB_TSL
        "pHSL": -0.25,
        "pPF_1": 0.023,
        "pPF_2": 0.042,
        "pSL_1": 0.018,
        "pSL_2": 0.041,

        "sell_1": True,
        "sell_2": True,
        "sell_3": True,
        "sell_4": True,
        "sell_5": True,
    }

    stoploss = -0.25
    timeframe = '5m'

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Custom stoploss
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 240

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

    # hard stoploss profit
    pHSL = DecimalParameter(-0.990, -0.040, default=-0.08, decimals=3, space='sell')
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.050, default=0.016, decimals=3, space='sell')
    pSL_1 = DecimalParameter(0.008, 0.050, default=0.011, decimals=3, space='sell')

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell')
    pSL_2 = DecimalParameter(0.040, 0.100, default=0.040, decimals=3, space='sell')

    sell_1 = CategoricalParameter([True, False], default=True, space="sell")
    sell_2 = CategoricalParameter([True, False], default=True, space="sell")
    sell_3 = CategoricalParameter([True, False], default=True, space="sell")
    sell_4 = CategoricalParameter([True, False], default=True, space="sell")
    sell_5 = CategoricalParameter([True, False], default=True, space="sell")

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

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = numpy.nan_to_num(ta.RSI(dataframe, timeperiod=5))
        rsiframe = DataFrame(dataframe['rsi']).rename(columns={'rsi': 'close'})
        dataframe['emarsi'] = numpy.nan_to_num(ta.EMA(rsiframe, timeperiod=5))
        dataframe['adx'] = numpy.nan_to_num(ta.ADX(dataframe))
        dataframe['minusdi'] = numpy.nan_to_num(ta.MINUS_DI(dataframe))
        minusdiframe = DataFrame(dataframe['minusdi']).rename(columns={'minusdi': 'close'})
        dataframe['minusdiema'] = numpy.nan_to_num(ta.EMA(minusdiframe, timeperiod=25))
        dataframe['plusdi'] = numpy.nan_to_num(ta.PLUS_DI(dataframe))
        plusdiframe = DataFrame(dataframe['plusdi']).rename(columns={'plusdi': 'close'})
        dataframe['plusdiema'] = numpy.nan_to_num(ta.EMA(plusdiframe, timeperiod=5))
        dataframe['lowsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=60))
        dataframe['highsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=120))
        dataframe['fastsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=120))
        dataframe['slowsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=240))
        dataframe['bigup'] = dataframe['fastsma'].gt(dataframe['slowsma']) & ((dataframe['fastsma'] - dataframe['slowsma']) > dataframe['close'] / 300)
        dataframe['bigdown'] = ~dataframe['bigup']
        dataframe['trend'] = dataframe['fastsma'] - dataframe['slowsma']
        dataframe['preparechangetrend'] = dataframe['trend'].gt(dataframe['trend'].shift())
        dataframe['preparechangetrendconfirm'] = dataframe['preparechangetrend'] & dataframe['trend'].shift().gt(dataframe['trend'].shift(2))
        dataframe['continueup'] = dataframe['slowsma'].gt(dataframe['slowsma'].shift()) & dataframe['slowsma'].shift().gt(dataframe['slowsma'].shift(2))
        dataframe['delta'] = dataframe['fastsma'] - dataframe['fastsma'].shift()
        dataframe['slowingdown'] = dataframe['delta'].lt(dataframe['delta'].shift())

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            dataframe['slowsma'].gt(0) &
            dataframe['close'].lt(dataframe['highsma']) &
            dataframe['close'].lt(dataframe['lowsma']) &
            dataframe['minusdi'].gt(dataframe['minusdiema']) &
            dataframe['rsi'].ge(dataframe['rsi'].shift()) &
            (
              (
                ~dataframe['preparechangetrend'] &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(25) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(20)
              ) |
              (
                ~dataframe['preparechangetrend'] &
                dataframe['continueup'] &
                dataframe['adx'].gt(30) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(20)
              ) |
              (
                ~dataframe['continueup'] &
                dataframe['adx'].gt(35) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(20)
              ) |
              (
                dataframe['continueup'] &
                dataframe['adx'].gt(30) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(25)
              )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(), 'sell'] = 0
        return dataframe

    def custom_sell(self, pair: str, trade: Trade, current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        if current_profit >= 0.023:
            return None

        if self.sell_1.value:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['close'] > last_candle['lowsma'] or last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['bigdown'])
            ):
                return "sell_1"

        if self.sell_2.value:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['emarsi'] > 75 or last_candle['close'] > last_candle['slowsma'])
                    and (last_candle['bigdown'])
            ):
                return "sell_2"

        if self.sell_3.value:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['adx'] > 30)
                    and (last_candle['emarsi'] >= 80)
                    and (last_candle['bigup'])
            ):
                return "sell_3"

        if self.sell_4.value:
            if (
                    (last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['slowingdown'])
                    and (last_candle['emarsi'] >= 75)
                    and (last_candle['slowsma'] > 0)
            ):
                return "sell_4"

        if self.sell_5.value:
            if (
                    (last_candle['preparechangetrendconfirm'])
                    and (last_candle['minusdi'] < last_candle['plusdi'])
                    and (last_candle['close'] > last_candle['lowsma'])
                    and (last_candle['slowsma'] > 0)
            ):
                return "sell_5"
