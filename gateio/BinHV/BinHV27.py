from datetime import datetime

from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, DecimalParameter, stoploss_from_open
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib

import talib.abstract as ta
import numpy  # noqa

TMP_HOLD = []


class BinHV27(IStrategy):
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
        # custom stop loss params
        "pHSL": -0.25,
        "pPF_1": 0.02,
        "pPF_2": 0.05,
        "pSL_1": 0.017,
        "pSL_2": 0.04
    }

    stoploss = -1
    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 240

    # default False
    use_custom_stoploss = False

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

    # buy params
    buy_adx1 = IntParameter(low=10, high=100, default=25, space='buy', optimize=True)
    buy_emarsi1 = IntParameter(low=10, high=100, default=20, space='buy', optimize=True)
    buy_adx2 = IntParameter(low=20, high=100, default=30, space='buy', optimize=True)
    buy_emarsi2 = IntParameter(low=20, high=100, default=20, space='buy', optimize=True)
    buy_adx3 = IntParameter(low=10, high=100, default=35, space='buy', optimize=True)
    buy_emarsi3 = IntParameter(low=10, high=100, default=20, space='buy', optimize=True)
    buy_adx4 = IntParameter(low=20, high=100, default=30, space='buy', optimize=True)
    buy_emarsi4 = IntParameter(low=20, high=100, default=25, space='buy', optimize=True)

    # trailing stoploss
    pHSL = DecimalParameter(-0.990, -0.040, default=-0.08, decimals=3, space='sell')
    pPF_1 = DecimalParameter(0.008, 0.050, default=0.016, decimals=3, space='sell')
    pSL_1 = DecimalParameter(0.008, 0.050, default=0.011, decimals=3, space='sell')
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell')
    pSL_2 = DecimalParameter(0.040, 0.100, default=0.040, decimals=3, space='sell')

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

        return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)

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

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
                            dataframe['adx'].gt(self.buy_adx1.value) &
                            dataframe['bigdown'] &
                            dataframe['emarsi'].le(self.buy_emarsi1.value)
                    ) |
                    (
                            ~dataframe['preparechangetrend'] &
                            dataframe['continueup'] &
                            dataframe['adx'].gt(self.buy_adx2.value) &
                            dataframe['bigdown'] &
                            dataframe['emarsi'].le(self.buy_emarsi2.value)
                    ) |
                    (
                            ~dataframe['continueup'] &
                            dataframe['adx'].gt(self.buy_adx3.value) &
                            dataframe['bigup'] &
                            dataframe['emarsi'].le(self.buy_emarsi3.value)
                    ) |
                    (
                            dataframe['continueup'] &
                            dataframe['adx'].gt(self.buy_adx4.value) &
                            dataframe['bigup'] &
                            dataframe['emarsi'].le(self.buy_emarsi4.value)
                    )
            ),
            ['enter_long', 'enter_tag']] = (1, 'long_in')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        previous_candle_1 = dataframe.iloc[-2].squeeze()

        # if use custom stoploss open it
        # if current_profit >= self.pPF_1.value:
        #     return None

        if (
                (last_candle['close'] > last_candle['bb_middleband'])
        ):
            if trade.id not in TMP_HOLD:
                TMP_HOLD.append(trade.id)
                return None

        for i in TMP_HOLD:
            if trade.id == i and (last_candle['close'] < previous_candle_1['close']):
                TMP_HOLD.remove(i)
                return "sell_drop"

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return 1.0
