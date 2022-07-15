"""
OnlyProfitHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss

# Optimize around the "average profit" result
EXPECTED_AVG_PROFIT = 2.0


class AvgProfitHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation takes only avg profit into account.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results.
        """

        profit_mean = results['profit_ratio'].mean() if len(results) > 0 else 0.0,
        profit_mean_pct = results['profit_ratio'].mean() * 100.0 if len(results) > 0 else 0.0,

        return 1 - profit_mean_pct / EXPECTED_AVG_PROFIT
