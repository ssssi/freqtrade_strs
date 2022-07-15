from datetime import datetime
from math import exp

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


# Define some constants:

# set TARGET_RATIO to your desired ratio of winning:losing trades
# a value of 10 would look for a win rate of 10:1, e.g. in 100 trades only 10 are losing.
TARGET_RATIO = 10

class WinRateHyperOptLoss(IHyperOptLoss):

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results
        """

        wins = len(results[results['profit_abs'] > 0])
        drawss = len(results[results['profit_abs'] == 0])
        losses = len(results[results['profit_abs'] < 0])

        winrate = wins / losses

        return 1 - winrate / TARGET_RATIO
