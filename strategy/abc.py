import mbt
import numpy as np
import alpha as al

class Strategy(mbt.Strategy):
  def execute(self, ctx: mbt.Context):
    """
    ABC 趋势回调策略
    
    策略逻辑：
    1. 趋势判断：MA20 > MA60 为上升趋势。
    2. 买入条件：上升趋势中，经历跌破 MA20 的回调后，收盘价重新站上 MA20 且放量。
    3. 卖出条件：
        - 固定止损：-6%
        - 移动止盈：最高点回落 -8%
        - 趋势破坏：跌破 MA60
    """
    close = ctx.data('close')
    volume = ctx.data('volume')
    ma20 = al.MA(close, 20)
    ma60 = al.MA(close, 60)
    
    # 1. 趋势判断
    uptrend = ma20 > ma60
    
    # 2. 买入条件
    # 前一交易日收盘价低于 MA20 (回调)
    pullback = al.REF(close < ma20, 1)
    # 当前交易日重新站上 MA20，且放量
    breakback = (close > ma20) & (volume > al.REF(volume, 1))
    
    buy_signals = uptrend & pullback & breakback
    
    # 3. 卖出条件
    ror_hold = ctx.data('ror_hold')
    
    # 固定止损：买入价下方-6%
    stop_loss = ror_hold <= -0.06
    
    # 移动止盈：持仓最高点回落-8%
    # 计算距离最近一次买入信号的 K 线数
    bars_since_buy = al.BARSLAST(buy_signals)
    # 获取持仓期间(从 buy_signal 触发至今)的最高收益率
    peak_ror = al.HHV(ror_hold, bars_since_buy + 1)
    # 回落计算: (1 + current_ror) / (1 + peak_ror) <= 0.92 (即从高点下跌 8%)
    trailing_stop = ((1 + ror_hold) / (1 + peak_ror)) <= 0.92
    
    # 趋势破坏：收盘价跌破MA60
    trend_broken = close < ma60
    
    # 只有在持仓时(ror_hold 不为 NaN)卖出信号才有效
    is_holding = ~np.isnan(ror_hold)
    sell_signals = is_holding & (stop_loss | trailing_stop | trend_broken)
    
    # 执行交易
    ctx.buy(buy_signals, 1.0)
    ctx.sell(sell_signals, 1.0)
