from datetime import datetime

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, DecimalParameter, stoploss_from_open
from freqtrade.strategy.interface import IStrategy
from technical.consensus import Consensus


class ConsensusShort(IStrategy):
    """
    come from https://github.com/werkkrew/freqtrade-strategies/blob/main/strategies/archived/consensus_strat.py
    Author:werkkrew
    """
    minimal_roi = {
        "0": 1,
        "120": 0
    }

    buy_params = {
        'buy_score_short': 20,

        # leverage set
        "leverage_num": 1,

    }

    sell_params = {
        'sell_score_short': 45,

        # custom stop loss params
        "pHSL": -0.25,
        "pPF_1": 0.012,
        "pPF_2": 0.05,
        "pSL_1": 0.01,
        "pSL_2": 0.04
    }

    stoploss = -0.99
    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count: int = 30

    can_short = True

    use_custom_stoploss = True

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

    buy_optimize = False
    buy_score_short = IntParameter(low=0, high=100, default=20, space='buy', optimize=buy_optimize)

    leverage_optimize = True
    leverage_num = IntParameter(low=1, high=20, default=1, space='buy', optimize=leverage_optimize)

    # trailing stoploss
    trailing_optimize = True
    pHSL = DecimalParameter(-0.990, -0.040, default=-0.08, decimals=3, space='sell', optimize=trailing_optimize)
    pPF_1 = DecimalParameter(0.008, 0.100, default=0.016, decimals=3, space='sell', optimize=trailing_optimize)
    pSL_1 = DecimalParameter(0.008, 0.100, default=0.011, decimals=3, space='sell', optimize=trailing_optimize)
    pPF_2 = DecimalParameter(0.040, 0.200, default=0.080, decimals=3, space='sell', optimize=trailing_optimize)
    pSL_2 = DecimalParameter(0.040, 0.200, default=0.040, decimals=3, space='sell', optimize=trailing_optimize)

    sell_optimize = False
    sell_score_short = IntParameter(low=0, high=100, default=45, space='sell', optimize=sell_optimize)

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Consensus strategy
        # add c.evaluate_indicator bellow to include it in the consensus score (look at
        # consensus.py in freqtrade technical)
        # add custom indicator with c.evaluate_consensus(prefix=<indicator name>)
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
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['consensus_buy'] < self.buy_score_short.value) &
                    (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        dataframe.loc[
            (
            ),
            'enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['consensus_sell'] < self.sell_score_short.value) &
                    (dataframe['volume'] > 0)
            ),
            'exit_short'] = 1

        dataframe.loc[
            (
            ),
            'exit_long'] = 0

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value

