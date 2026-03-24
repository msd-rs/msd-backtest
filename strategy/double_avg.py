import mbt
import numpy as np
import alpha as al

class Strategy(mbt.Strategy):
  def __init__(self, fast_period: int = 5, slow_period: int = 20):
    super().__init__()
    self.fast_period = fast_period
    self.slow_period = slow_period

  def execute(self, ctx: mbt.Context):
    """
    execute the strategy, this method will be called for each bar.

    Args:
      ctx: context of the strategy 
    """
    close_price = ctx.data('close')
    
    # Calculate fast and slow moving averages
    fast_ma = al.MA(close_price, self.fast_period)
    slow_ma = al.MA(close_price, self.slow_period)
    
    # Build buy and sell signals
    buy_signals = al.CROSS(fast_ma, slow_ma)
    sell_signals = al.RCROSS(fast_ma, slow_ma)
    
    # Execute trades
    ctx.buy(buy_signals, 1.0)
    ctx.sell(sell_signals, 1.0)
