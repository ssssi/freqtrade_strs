from datetime import datetime
from functools import reduce
import numpy as np
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter, merge_informative_pair, stoploss_from_open
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta

class Optimized_BinHV27(IStrategy):
    """
    优化版多空混合策略
    改进点：
    1. 增加15分钟和1小时时间框架的协同验证
    2. 动态参数调整机制
    3. ATR波动率感知止损
    4. 杠杆动态管理
    5. 条件逻辑简化
    """
    
    # 基础配置
    timeframe = '5m'
    inf_timeframes = ['4h', '1h', '15m']  # 新增多时间框架
    stoploss = -0.99
    process_only_new_candles = True
    startup_candle_count = 480  # 适应多时间框架
    
    # 参数优化空间（示例）
    buy_params = {
        'entry_adx': 28,
        'entry_emarsi': 22,
        'volatility_threshold': 0.03
    }
    
    # 动态参数定义
    entry_adx = IntParameter(20, 40, default=28, space='buy')
    entry_emarsi = IntParameter(15, 30, default=22, space='buy')
    volatility_threshold = DecimalParameter(0.02, 0.05, default=0.03, decimals=3, space='sell')
    
    # 新增波动率指标
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 原始指标计算
        dataframe['rsi'] = ta.RSI(dataframe, 5)
        dataframe['emarsi'] = ta.EMA(dataframe['rsi'], 5)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['atr'] = ta.ATR(dataframe, 14)  # 新增ATR
        
        # 多时间框架协同
        for tf in self.inf_timeframes:
            inf_data = self.dp.get_pair_dataframe(metadata['pair'], tf)
            
            # 趋势方向判断
            inf_data['hlc3'] = (inf_data['high'] + inf_data['low'] + inf_data['close']) / 3
            inf_data['tsf'] = ta.TSF(inf_data['hlc3'], 2)
            
            # 趋势强度指标
            inf_data[f'trend_strength_{tf}'] = ta.EMA(inf_data['close'], 20) - ta.EMA(inf_data['close'], 50)
            
            # 合并数据
            dataframe = merge_informative_pair(
                dataframe, inf_data, self.timeframe, tf, 
                suffixes=("", f"_{tf}")
            )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        优化后的入场逻辑：
        1. 多时间框架趋势一致性验证
        2. 波动率过滤
        3. 简化条件组合
        """
        # 多时间框架趋势确认
        trend_confirm = (
            (dataframe['trend_strength_4h'] > 0) &
            (dataframe['trend_strength_1h'] > 0) &
            (dataframe['trend_strength_15m'] > 0)
        )
        
        # 波动率过滤
        volatility_ok = dataframe['atr']/dataframe['close'] < self.volatility_threshold.value
        
        # 统一入场条件
        long_condition = (
            trend_confirm &
            volatility_ok &
            (dataframe['adx'] > self.entry_adx.value) &
            (dataframe['emarsi'] < self.entry_emarsi.value) &
            (dataframe['close'] < dataframe['ema_20'])
        )
        
        short_condition = (
            (~trend_confirm) &
            volatility_ok &
            (dataframe['adx'] > self.entry_adx.value) &
            (dataframe['emarsi'] > (100 - self.entry_emarsi.value)) &
            (dataframe['close'] > dataframe['ema_20'])
        )
        
        dataframe.loc[long_condition, 'enter_long'] = 1
        dataframe.loc[short_condition, 'enter_short'] = 1
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        波动率感知动态止损：
        1. 基础止损：开仓价的2.5ATR
        2. 盈利保护：浮盈超过5%后启用追踪止损
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe['atr'].iloc[-1]
        
        # 基础止损
        if trade.is_short:
            base_sl = current_rate + 2.5 * atr
        else:
            base_sl = current_rate - 2.5 * atr
        
        # 盈利保护机制
        if current_profit > 0.05:
            trail_sl = current_profit * 0.5  # 保留50%利润
            return stoploss_from_open(trail_sl, current_profit, is_short=trade.is_short)
        
        return stoploss_from_open(base_sl, current_profit, is_short=trade.is_short)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        动态杠杆管理：
        1. 高波动率时降低杠杆
        2. 趋势强度影响杠杆倍数
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        volatility = dataframe['atr'].iloc[-1] / dataframe['close'].iloc[-1]
        trend_strength = dataframe['trend_strength_4h'].iloc[-1]
        
        # 波动率控制
        if volatility > 0.04:
            return 1
        
        # 趋势强度加成
        leverage = min(
            3, 
            1 + abs(trend_strength) * 10  # 趋势强度每增加0.1，杠杆加0.1
        )
        
        return leverage

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        简化退出逻辑：
        1. 趋势强度衰减退出
        2. 波动率突破退出
        """
        # 趋势衰减信号
        trend_decay = (
            (dataframe['trend_strength_4h'] < 0) |
            (dataframe['trend_strength_1h'] < 0.5 * dataframe['trend_strength_4h'])
        )
        
        # 波动率突破
        volatility_break = dataframe['atr']/dataframe['close'] > self.volatility_threshold.value * 1.5
        
        dataframe.loc[trend_decay | volatility_break, 'exit_long'] = 1
        dataframe.loc[trend_decay | volatility_break, 'exit_short'] = 1
        
        return dataframe
