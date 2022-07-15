"""
WinHyperOptLoss

This module is a custom HyperoptLoss class based on Profit and Win/Loss ratio

To deploy this, copy the file to the <freqtrade>/user_data/hyperopts directory
"""
from math import exp

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss
from datetime import datetime
import numpy as np
from typing import Any, Dict


# Contstants to allow evaluation in cases where thre is insufficient (or nonexistent) info in the configuration
EXPECTED_TRADES_PER_DAY = 1                         # used to set target goals
MIN_TRADES_PER_DAY = EXPECTED_TRADES_PER_DAY / 8    # used to filter out scenarios where there are not enough trades
UNDESIRED_SOLUTION = 2.0             # indicates that we don't want this solution (so hyperopt will avoid)


class WinHyperOptLoss(IHyperOptLoss):


    """
    Defines a custom loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               backtest_stats: Dict[str, Any],
                               *args, **kwargs) -> float:

        debug_level = 1 # displays (more) messages if higher

        days_period = (max_date - min_date).days
        # target_trades = days_period*EXPECTED_TRADES_PER_DAY
        if config['max_open_trades']:
            target_trades = days_period * config['max_open_trades']
        else:
            target_trades = days_period * EXPECTED_TRADES_PER_DAY

        # Calculate trade loss metric first, because this is used elsewhere
        # Several other metrics are misleading if there are not enough trades

        # trade loss
        if trade_count > MIN_TRADES_PER_DAY * days_period:
            num_trades_loss = (target_trades - trade_count) / target_trades
        else:
            # just return a large number if insufficient trades. Makes other calculations easier/safer
            if debug_level > 1:
                print(" \tTrade count too low:{:.0f}".format(trade_count))
            return UNDESIRED_SOLUTION


        # Winning trades

        total_profit = results["profit_abs"]

        if backtest_stats['wins']:
            winning_count = backtest_stats['wins']
        else:
            results['upside_returns'] = 0
            results.loc[total_profit > 0.0001, 'upside_returns'] = 1.0
            winning_count = results['upside_returns'].sum()

        # calculate win ratio loss. Scale so that 0.0 equates to 50% win/loss ratio
        win_ratio_loss = 10.0 * (0.5 - winning_count / trade_count)

        # use profit and drawdown as a tie-breaker

        # profitable?
        profit_loss = -backtest_stats['profit_total']
        if profit_loss > 0:
            # print("profit: {:.2f}".format(profit_loss))
            return UNDESIRED_SOLUTION + abs(profit_loss)

        drawdown_loss = 0.0
        if 'max_drawdown' in backtest_stats:
            drawdown_loss = (backtest_stats['max_drawdown'] - 1.0)

        result = win_ratio_loss + drawdown_loss + profit_loss

        return result

