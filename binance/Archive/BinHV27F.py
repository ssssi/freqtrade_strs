from datetime import datetime
from functools import reduce

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter, merge_informative_pair, stoploss_from_open
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib

import talib.abstract as ta
import numpy  # noqa

class BinHV27_combined(IStrategy):
    """
    BinHV27组合策略（多空混合版）
    源码整合自多个BinHV27策略文件，通过TSF指标动态切换多空方向
    策略特点：
    - 结合4小时时间框架的TSF指标判断趋势方向
    - 使用ADX、EMA-RSI、双均线系统等多维度技术指标
    - 支持做多做空双向交易
    - 动态止盈止损机制
    - 可配置杠杆倍数
    """

    # 最小ROI设置（粗暴式设置，实际通过动态止损实现退出）
    minimal_roi = {"0": 1}

    # 策略参数默认值（通过优化得出）
    buy_params = {...}  # 原始参数值已省略
    sell_params = {...}  # 原始参数值已省略

    # 基础策略配置
    stoploss = -0.99       # 初始止损（实际使用动态止损覆盖）
    timeframe = '5m'       # 主要交易时间框架
    inf_timeframe = '4h'   # 趋势判断时间框架
    process_only_new_candles = True
    startup_candle_count = 240  # 启动所需K线数（4小时框架需要240根5分钟K线）

    # 启用高级功能
    use_custom_stoploss = True  # 使用动态止损
    can_short = True            # 允许做空

    #====================== 参数定义 ======================
    # 注：以下参数通过超参优化得出，空间分为buy/sell，部分参数已固定
    
    #--------- 入场参数 ---------
    entry_optimize = True  # 是否优化入场参数
    # 多头入场ADX阈值（四组不同条件）
    entry_long_adx1 = IntParameter(10, 100, default=25, space='buy', optimize=entry_optimize)
    # 多头入场EMA-RSI阈值
    entry_long_emarsi1 = IntParameter(10, 100, default=20, space='buy', optimize=entry_optimize)
    ...  # 其他类似参数省略

    #--------- 动态止损参数 ---------
    trailing_optimize = False  # 是否优化止损参数（默认关闭）
    # 多头硬止损阈值
    pHSL_long = DecimalParameter(-0.99, -0.04, default=-0.08, decimals=3, space='sell', optimize=trailing_optimize)
    ...  # 其他类似参数省略

    #--------- 离场参数 ---------
    exit_optimize = True  # 是否优化离场参数
    # 多头离场ADX阈值
    exit_long_adx2 = IntParameter(10, 100, default=30, space='sell', optimize=exit_optimize)
    ...  # 其他类似参数省略

    #--------- 开关类参数 ---------
    exit2_optimize = True  # 是否优化退出条件开关
    # 启用/禁用特定退出条件
    exit_long_1 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)
    ...  # 其他类似参数省略

    #--------- 杠杆参数 ---------
    leverage_optimize = False
    leverage_num = IntParameter(1, 5, default=1, space='sell', optimize=leverage_optimize)

    #====================== 核心逻辑 ======================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算技术指标"""
        # 核心指标计算
        dataframe['rsi'] = ta.RSI(dataframe, 5)                   # 5周期RSI
        dataframe['emarsi'] = ta.EMA(DataFrame(dataframe['rsi']), 5)  # RSI的5周期EMA
        dataframe['adx'] = ta.ADX(dataframe)                      # 平均趋向指数
        dataframe['minusdi'] = ta.MINUS_DI(dataframe)             # 负方向指标
        dataframe['plusdi'] = ta.PLUS_DI(dataframe)               # 正方向指标
        
        # 均线系统
        dataframe['lowsma'] = ta.EMA(dataframe, 60)               # 60周期EMA（短期）
        dataframe['highsma'] = ta.EMA(dataframe, 120)             # 120周期EMA（中期）
        dataframe['fastsma'] = ta.SMA(dataframe, 120)             # 120周期SMA
        dataframe['slowsma'] = ta.SMA(dataframe, 240)             # 240周期SMA（长期）
        
        # 趋势判断指标
        dataframe['bigup'] = dataframe['fastsma'] > dataframe['slowsma']  # 快线在慢线上方
        dataframe['trend'] = dataframe['fastsma'] - dataframe['slowsma']  # 趋势强度
        dataframe['preparechangetrend'] = dataframe['trend'] > dataframe['trend'].shift()  # 趋势反转信号
        
        # 布林带指标
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']  # 布林带下轨
        
        # 4小时框架趋势判断指标
        inf_dataframe = self.dp.get_pair_dataframe(metadata['pair'], self.inf_timeframe)
        inf_dataframe['hlc3'] = ta.TYPPRICE(inf_dataframe)        # 典型价格（高+低+收）/3
        inf_dataframe['tsf'] = ta.TSF(inf_dataframe['hlc3'], 2)  # 时间序列预测指标
        inf_dataframe['allow_long'] = inf_dataframe['tsf'] / inf_dataframe['hlc3'] > 1.01  # 允许做多信号
        inf_dataframe['allow_short'] = inf_dataframe['tsf'] / inf_dataframe['hlc3'] < 0.99 # 允许做空信号
        
        # 合并多时间框架数据
        dataframe = merge_informative_pair(dataframe, inf_dataframe, self.timeframe, self.inf_timeframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """生成入场信号"""
        # 多头入场条件（四组不同参数组合）
        long_entry_1 = (
            dataframe[f'allow_long_{self.inf_timeframe}'] &  # 4小时框架允许做多
            (dataframe['close'] < dataframe['lowsma']) &     # 价格低于短期均线
            (dataframe['adx'] > self.entry_long_adx1.value) &# ADX超过阈值
            (dataframe['emarsi'] <= self.entry_long_emarsi1.value)  # EMA-RSI低于阈值
            ...  # 其他条件省略
        )
        # 类似条件组合long_entry_2/3/4...

        # 空头入场条件（与多头逻辑镜像）
        short_entry_1 = (
            dataframe[f'allow_short_{self.inf_timeframe}'] &  # 4小时框架允许做空
            (dataframe['close'] < dataframe['lowsma']) &      # 价格低于短期均线
            (dataframe['adx'] > self.entry_short_adx1.value) & 
            ...  # 其他条件省略
        )
        # 类似条件组合short_entry_2/3/4...

        # 组合所有入场条件
        dataframe.loc[long_conditions, 'enter_long'] = 1
        dataframe.loc[short_conditions, 'enter_short'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        """自定义离场逻辑"""
        # 动态止盈止损检查
        if current_profit >= self.pPF_1_long.value and not trade.is_short:
            return None  # 达到一级止盈点但继续持有
            
        # 多头退出条件1：趋势反转确认
        if self.exit_long_1.value and not trade.is_short:
            if (last_candle['close'] > last_candle['highsma']) and (last_candle['bigdown']):
                return "exit_long_1"  # 价格突破均线且趋势转弱
        
        # 类似的其他退出条件判断...
        
        return None  # 默认不触发退出

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        """动态止损计算"""
        # 根据持仓方向选择参数
        if trade.is_short:
            hsl = self.pHSL_short.value
            pf1 = self.pPF_1_short.value
        else:
            hsl = self.pHSL_long.value
            pf1 = self.pPF_1_long.value
        
        # 动态计算止损位：盈利越高，止损越宽松
        if current_profit > pf1:
            ...  # 计算线性插值止损
        return stoploss_from_open(...)  # 最终止损位计算

    def leverage(self, pair: str, current_time: datetime, current_rate: float, 
                proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """杠杆控制"""
        return self.leverage_num.value  # 返回配置的杠杆倍数

# 策略核心机制说明：
# 1. 多时间框架协同：5分钟线用于交易信号，4小时线通过TSF指标判断趋势方向
# 2. 入场逻辑：结合ADX趋势强度、EMA-RSI超卖超买状态、均线位置等多重过滤
# 3. 动态风控：采用分段止盈止损，盈利越高允许的回撤空间越大
# 4. 趋势跟踪：通过双均线系统（120/240周期）识别长期趋势
# 5. 双向交易：根据4小时TSF指标自动切换多空方向
