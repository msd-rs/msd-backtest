import numpy as np
from mbt import PortfolioStrategy, PortfolioAccount, PortfolioRunner, DataProvider
from mbt.portfolio import PortfolioContext


def test_rebalance_and_sell_conflict_prioritizes_sell():
  """
  Verify that when a strategy calls both ctx.sell(obj) (e.g. risk control)
  and ctx.rebalance(weights) on the same bar, the sell order is executed first,
  its target weight is reduced to prevent immediate rebuy, and the sell is NOT dropped.
  """
  dp = DataProvider(
    {
      "open": np.ones(10),
      "close": np.ones(10),
      "price": np.ones(10),
      "ts": np.tile(
        np.array(
          [
            "2024-01-01T00:00:00.000000",
            "2024-01-02T00:00:00.000000",
            "2024-01-03T00:00:00.000000",
            "2024-01-04T00:00:00.000000",
            "2024-01-05T00:00:00.000000",
          ],
          dtype="datetime64[us]",
        ),
        2,
      ),
    },
    objs=["SH600000", "SZ000001"],
  )

  class ConflictStrategy(PortfolioStrategy):
    def execute(self, ctx: PortfolioContext):
      if ctx.bar == 1:
        # Initial buy into SH600000
        ctx.rebalance(np.array([0.5, 0.0]))
      elif ctx.bar == 2:
        # On bar 2, emergency sell SH600000 AND simultaneously rebalance
        # Even though target_weights still specifies 0.5 for SH600000, the sell must take priority!
        ctx.sell(np.array([True, False]), percent=1.0)
        ctx.rebalance(np.array([0.5, 0.5]))

  acc = PortfolioAccount(100000.0, dp)
  runner = PortfolioRunner(acc)
  runner.run(ConflictStrategy())

  # Bar 1: bought SH600000
  # Bar 2: SH600000 was sold via ctx.sell(percent=1.0) and target weight for SH600000 was zeroed out.
  # SZ000001 was bought with the target weight of 0.5
  assert acc.positions[0].shares == 0.0  # SH600000 sold out!
  assert acc.positions[1].shares > 0.0   # SZ000001 bought!
  # Verify records contain the sell
  sell_records = [r for r in acc.records if r.bar == 2 and r.action == "SELL"]
  assert len(sell_records) == 1
  assert sell_records[0].group == 0
