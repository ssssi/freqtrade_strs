"""
MarketHyperOptLoss

This module is a custom HyperoptLoss class based on performance relative to the overall market

To deploy this, copy the file to the <freqtrade>/user_data/hyperopts directory
"""
from math import exp

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss
from datetime import datetime
import numpy as np
from typing import Any, Dict


# Contstants to allow evaluation in cases where thre is insufficient (or nonexistent) info in the configuration
EXPECTED_TRADES_PER_DAY = 2                         # used to set target goals
MIN_TRADES_PER_DAY = EXPECTED_TRADES_PER_DAY / 8    # used to filter out scenarios where there are not enough trades
UNDESIRED_SOLUTION = 2.0             # indicates that we don't want this solution (so hyperopt will avoid)


class MarketHyperOptLoss(IHyperOptLoss):


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


        # Compare to the overall market performance

        total_profit = results["profit_abs"]
        if ('market_change' in results):
            market_profit = results["market_change"]
        elif ('market_change' in backtest_stats):
            market_profit = backtest_stats["market_change"]
        else:
            print("Market performance not available")
            market_profit = results["profit_abs"]

        market_loss = 10.0 * (market_profit - total_profit)

        # use drawdown as a tie-breaker
        drawdown_loss = 0.0
        if backtest_stats['max_drawdown']:
            drawdown_loss = (backtest_stats['max_drawdown'] - 1.0)

        result = market_loss + drawdown_loss

        return result

