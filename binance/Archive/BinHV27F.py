from datetime import datetime
from functools import reduce

from freqtrade.strategy import IntParameter, CategoricalParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

import talib.abstract as ta
import numpy  # noqa


class BinHV27F(IStrategy):
    """

        strategy sponsored by user BinH from slack

    """

    minimal_roi = {
        "0": 100
    }

    buy_params = {
        'buy_adx1': 25,
        'buy_emarsi1': 20,
        'buy_adx2': 30,
        'buy_emarsi2': 20,
        'buy_adx3': 35,
        'buy_emarsi3': 20,
        'buy_adx4': 30,
        'buy_emarsi4': 25
    }

    sell_params = {
        # sell params
        'emarsi1': 75,
        'adx2': 30,
        'emarsi2': 80,
        'emarsi3': 75
    }

    stoploss = -0.25
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
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # buy params
    buy_adx1 = IntParameter(low=10, high=100, default=25, space='buy', optimize=True)
    buy_emarsi1 = IntParameter(low=10, high=100, default=20, space='buy', optimize=True)
    buy_adx2 = IntParameter(low=20, high=100, default=30, space='buy', optimize=True)
    buy_emarsi2 = IntParameter(low=20, high=100, default=20, space='buy', optimize=True)
    buy_adx3 = IntParameter(low=10, high=100, default=35, space='buy', optimize=True)
    buy_emarsi3 = IntParameter(low=10, high=100, default=20, space='buy', optimize=True)
    buy_adx4 = IntParameter(low=20, high=100, default=30, space='buy', optimize=True)
    buy_emarsi4 = IntParameter(low=20, high=100, default=25, space='buy', optimize=True)

    # buy_1_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    # buy_2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    # buy_3_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    # buy_4_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    #
    # sell_1_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    # sell_2_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    # sell_3_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    # sell_4_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    # sell_5_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True)

    leverage_optimize = False
    leverage_num = IntParameter(low=1, high=3, default=3, space='buy', optimize=leverage_optimize)

    # sell params
    adx2 = IntParameter(low=10, high=100, default=30, space='sell', optimize=True)
    emarsi1 = IntParameter(low=10, high=100, default=75, space='sell', optimize=True)
    emarsi2 = IntParameter(low=20, high=100, default=80, space='sell', optimize=True)
    emarsi3 = IntParameter(low=20, high=100, default=75, space='sell', optimize=True)

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
        dataframe['bigup'] = dataframe['fastsma'].gt(dataframe['slowsma']) & (
                    (dataframe['fastsma'] - dataframe['slowsma']) > dataframe['close'] / 300)
        dataframe['bigdown'] = ~dataframe['bigup']
        dataframe['trend'] = dataframe['fastsma'] - dataframe['slowsma']
        dataframe['preparechangetrend'] = dataframe['trend'].gt(dataframe['trend'].shift())
        dataframe['preparechangetrendconfirm'] = dataframe['preparechangetrend'] & dataframe['trend'].shift().gt(
            dataframe['trend'].shift(2))
        dataframe['continueup'] = dataframe['slowsma'].gt(dataframe['slowsma'].shift()) & dataframe[
            'slowsma'].shift().gt(dataframe['slowsma'].shift(2))
        dataframe['delta'] = dataframe['fastsma'] - dataframe['fastsma'].shift()
        dataframe['slowingdown'] = dataframe['delta'].lt(dataframe['delta'].shift())
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        buy_1 = (
                # self.buy_1_enable.value &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['preparechangetrend'] &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx1.value) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(self.buy_emarsi1.value)
        )

        buy_2 = (
                # self.buy_2_enable.value &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['preparechangetrend'] &
                dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx2.value) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(self.buy_emarsi2.value)
        )

        buy_3 = (
                # self.buy_3_enable.value &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx3.value) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(self.buy_emarsi3.value)
        )

        buy_4 = (
                # self.buy_4_enable.value &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx4.value) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(self.buy_emarsi4.value)
        )

        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        conditions.append(buy_2)
        dataframe.loc[buy_2, 'enter_tag'] += 'buy_2'

        conditions.append(buy_3)
        dataframe.loc[buy_3, 'enter_tag'] += 'buy_3'

        conditions.append(buy_4)
        dataframe.loc[buy_4, 'enter_tag'] += 'buy_4'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        s1 = (
                # self.sell_1_enable.value &
                ~dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                (dataframe['close'].gt(dataframe['lowsma']) | dataframe['close'].gt(dataframe['highsma'])) &
                dataframe['highsma'].gt(0) &
                dataframe['bigdown']
        )

        s2 = (
                # self.sell_2_enable.value &
                ~dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                dataframe['close'].gt(dataframe['highsma']) &
                dataframe['highsma'].gt(0) &
                (dataframe['emarsi'].ge(self.emarsi1.value) | dataframe['close'].gt(dataframe['slowsma'])) &
                dataframe['bigdown']
        )

        s3 = (
                # self.sell_3_enable.value &
                ~dataframe['preparechangetrendconfirm'] &
                dataframe['close'].gt(dataframe['highsma']) &
                dataframe['highsma'].gt(0) &
                dataframe['adx'].gt(self.adx2.value) &
                dataframe['emarsi'].ge(self.emarsi2.value) &
                dataframe['bigup']
        )

        s4 = (
                # self.sell_4_enable.value &
                dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                dataframe['slowingdown'] &
                dataframe['emarsi'].ge(self.emarsi3.value) &
                dataframe['slowsma'].gt(0)
        )

        s5 = (
                # self.sell_5_enable.value &
                dataframe['preparechangetrendconfirm'] &
                dataframe['minusdi'].lt(dataframe['plusdi']) &
                dataframe['close'].gt(dataframe['lowsma']) &
                dataframe['slowsma'].gt(0)
        )

        conditions.append(s1)
        dataframe.loc[s1, 'exit_tag'] += 's1 '

        conditions.append(s2)
        dataframe.loc[s2, 'exit_tag'] += 's2 '

        conditions.append(s3)
        dataframe.loc[s3, 'exit_tag'] += 's3 '

        conditions.append(s4)
        dataframe.loc[s4, 'exit_tag'] += 's4 '

        conditions.append(s5)
        dataframe.loc[s5, 'exit_tag'] += 's5 '

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value
