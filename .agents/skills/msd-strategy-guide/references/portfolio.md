# Portfolio Rebalancing Strategy (`mbt.PortfolioStrategy`)

When the strategy needs to manage a **multi-stock portfolio sharing a single cash pool** and rebalance positions across stocks (rather than testing each stock independently), subclass `mbt.PortfolioStrategy` and use `ctx.rebalance(weights)` (or `ctx.buy` / `ctx.sell`):

```python
class Strategy(mbt.PortfolioStrategy):

  def __init__(self, period: int = 20, top_k: int = 5):
    super().__init__()
    self.period = period
    self.top_k = top_k

  def prepare(self, ctx: mbt.Context):
    # Pre-compute momentum factor across all stocks
    self.ma = al.MA(ctx.close, self.period)

  def execute(self, ctx: mbt.Context):
    # Call ctx.rebalance(weights) on rebalance bars; sells are executed first to free cash for buys
    ...
```

Apart from the different base class and calling `rebalance` at the appropriate time, everything else can be referenced from [How to build a trading strategy](./trading.md)
