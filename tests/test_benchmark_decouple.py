import numpy as np
from mbt import DataProvider, PortfolioAccount, PortfolioRunner, PortfolioStrategy
from mbt.portfolio import PortfolioContext
from mbt.run.report import build_portfolio_report


def test_benchmark_not_at_index_zero():
  """Verify that benchmark obj at arbitrary index is properly marked untradable
  and correctly used for relative metrics calculation in build_portfolio_report.
  """
  bars = 5
  objs = ["SH600000", "SH000300", "SZ000001"]  # Benchmark SH000300 is at index 1
  # SH600000 flat, SH000300 doubles (+100%), SZ000001 flat
  prices = np.concatenate(
    [
      np.full(bars, 10.0),                     # SH600000
      np.linspace(10.0, 20.0, bars),           # SH000300 (benchmark)
      np.full(bars, 10.0),                     # SZ000001
    ]
  )
  dp = DataProvider(
    {
      "open": prices.copy(),
      "close": prices.copy(),
      "price": prices.copy(),
      "bw_price": prices.copy(),
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
        len(objs),
      ),
    },
    objs=objs,
  )

  acc = PortfolioAccount(100000.0, dp, benchmark="SH000300")
  assert acc.benchmark_idx == 1
  assert acc.tradable_mask[0] is True or acc.tradable_mask[0] == 1
  assert acc.tradable_mask[1] is False or acc.tradable_mask[1] == 0
  assert acc.tradable_mask[2] is True or acc.tradable_mask[2] == 1

  class SimpleBuyStrategy(PortfolioStrategy):
    def execute(self, ctx: PortfolioContext):
      if ctx.bar == 1:
        # Rebalance across all 3 assets with equal target
        ctx.rebalance(np.array([0.33, 0.33, 0.33]))

  runner = PortfolioRunner(acc)
  runner.run(SimpleBuyStrategy())

  # Group 1 (SH000300) must NOT have been bought!
  assert acc.positions[1].shares == 0.0
  assert acc.positions[0].shares > 0.0
  assert acc.positions[2].shares > 0.0

  reports = build_portfolio_report(dp, acc)
  assert len(reports) == 1
  # Benchmark profit should reflect SH000300 doubling (+1.0), not SH600000 flat (0.0)
  assert np.isclose(reports[0].benchmark_profit, 1.0, atol=0.01)
