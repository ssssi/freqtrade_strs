"""
ExpectancyHyperOptLoss

This module is a custom HyperoptLoss class

The goal is to use Expectancy as a metric, but also filters out bad scenarios (losing, not enough tradees etc)
For details on Expectancy, refere to: https://www.freqtrade.io/en/stable/edge/

To deploy this, copy the file to the <freqtrade>/user_data/hyperopts directory
"""
from math import exp

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss
from datetime import datetime
import numpy as np
from typing import Any, Dict


# Contstants to allow evaluation in cases where thre is insufficient (or nonexistent) info in the configuration
EXPECTED_TRADES_PER_DAY = 3                         # used to set target goals
MIN_TRADES_PER_DAY = EXPECTED_TRADES_PER_DAY / 3    # used to filter out scenarios where there are not enough trades
UNDESIRED_SOLUTION = 2.0             # indicates that we don't want this solution (so hyperopt will avoid)



class ExpectancyHyperOptLoss(IHyperOptLoss):


    """
    Defines a custom loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               backtest_stats: Dict[str, Any],
                               *args, **kwargs) -> float:

        debug_level = 0 # displays (more) messages if higher

        # if debug_level >= 2:
        #     profit_cols = [col for col in results.columns if 'profit' in col]
        #     print("Profit columns:")
        #     print(profit_cols)

        days_period = (max_date - min_date).days
        # target_trades = days_period*EXPECTED_TRADES_PER_DAY
        if config['max_open_trades']:
            target_trades = days_period * config['max_open_trades']
        else:
            target_trades = days_period * EXPECTED_TRADES_PER_DAY

        # Calculate trade loss metric first, because this is used elsewhere
        # Several other metrics are misleading if there are not enough trades

        # # trade loss
        # if trade_count > MIN_TRADES_PER_DAY * days_period:
        #     num_trades_loss = (target_trades - trade_count) / target_trades
        # else:
        #     # just return a large number if insufficient trades. Makes other calculations easier/safer
        #     if debug_level > 1:
        #         print(" \tTrade count too low:{:.0f}".format(trade_count))
        #     return UNDESIRED_SOLUTION

        stake = backtest_stats['stake_amount']
        total_profit_pct = results["profit_abs"] / stake

        # Winning trades
        results['upside_returns'] = 0
        results.loc[total_profit_pct > 0.0001, 'upside_returns'] = 1.0

        if backtest_stats['wins']:
            winning_count = backtest_stats['wins']
        else:
            winning_count = results['upside_returns'].sum()

        # Losing trades
        results['downside_returns'] = 0
        results.loc[total_profit_pct < 0, 'downside_returns'] = 1.0


        # Expectancy (refer to freqtrade edge page for info)
        w = winning_count / trade_count
        l = 1.0 - w
        results['net_gain'] = total_profit_pct * results['upside_returns']
        results['net_loss'] = total_profit_pct * results['downside_returns']
        ave_profit = results['net_gain'].sum() / trade_count
        ave_loss = results['net_loss'].sum() / trade_count

        if abs(ave_loss) < 0.01:
            ave_loss = 0.01  # set min loss = 1%, otherwise results can be wildly skewed
        r = ave_profit / abs(ave_loss)
        e = r * w - l

        # expectancy_loss = 1.0 - e  # goal is <1.0
        expectancy_loss = -e
        drawdown_loss = 0.0
        abs_profit_loss = 0.0

        # if (expectancy_loss <= 0.0):
        # use drawdown and profit as a tie-breaker
        if 'max_drawdown' in backtest_stats:
            drawdown_loss = (backtest_stats['max_drawdown'] - 1.0) / 2.0

        if 'profit_total' in backtest_stats:
            abs_profit_loss = -backtest_stats['profit_total'] / 2.0

        result = expectancy_loss + drawdown_loss + abs_profit_loss

        if ((debug_level == 1) & (result<0.0)) | (debug_level > 1):
            print("{:.2f} exp:{:.2f} drw:{:.2f} prf:{:.2f} ".format(result, expectancy_loss, drawdown_loss, abs_profit_loss))

        return result
