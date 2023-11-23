# freqtrade-strs
## some strategies.

# Don't try to create a complex strategy. Simple is often the most effective.

## Never set the stoploss to -0.99, it's very stupid in my opinion.
## Note:Don't use it directly(E0V1E.py), you need to optimize the parameters listed below, my dry run results are optimized after running, only need optimized these parameters, nothing else.

is_optimize_32 = True

buy_rsi_fast_32 = IntParameter(20, 70, default=45, space='buy', optimize=is_optimize_32)

buy_rsi_32 = IntParameter(15, 50, default=35, space='buy', optimize=is_optimize_32)

buy_sma15_32 = DecimalParameter(0.900, 1, default=0.961, decimals=3, space='buy', optimize=is_optimize_32)

buy_cti_32 = DecimalParameter(-1, 0, default=-0.58, decimals=2, space='buy', optimize=is_optimize_32)

sell_fastx = IntParameter(50, 100, default=75, space='sell', optimize=True)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## How do i hyper my str(current use is E0V1E.py).

SortinoHyperOptLossDaily for buy signal

SortinoHyperOptLoss for sell signal

## Note：which loss function to hyper this strategy is up to you. It's not set in stone

For help:

https://www.freqtrade.io/

Freqtrade official discord:

https://discord.gg/e8dkbJsKf5

if you have any question you can contact me in discord:@E0V1E or @evilzzq

