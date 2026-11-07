import numpy as np
import polars as pl
from mbt import (
  DataProvider,
  PortfolioAccount,
  PortfolioContext,
  PortfolioRunner,
  Selector,
  SelectorStrategy,
  Strategy,
)


def test_portfolio_shared_cash_rebalance():
  # 2 stocks, 4 bars:
  # Stock A: 10 -> 20 -> 20 -> 20 (doubles at bar 1)
  # Stock B: 10 -> 10 -> 10 -> 30 (triples at bar 3)
  prices_a = np.array([10.0, 20.0, 20.0, 20.0])
  prices_b = np.array([10.0, 10.0, 10.0, 30.0])
  prices = np.concatenate([prices_a, prices_b])

  dp = DataProvider(
    {
      "ts": np.tile(np.arange(4), 2),
      "open": prices,
      "close": prices,
      "price": prices,
      "dividend": np.zeros(8),
      "transfer_shares": np.zeros(8),
      "right_shares": np.zeros(8),
      "right_price": np.zeros(8),
    },
    objs=["SH600000", "SZ000001"],
  )

  class RotateStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):  # type: ignore[override]
      if ctx.dp.i == 1:
        # Bar 1: allocate 100% to Stock A (@ 20.0 -> 500 shares = 10000)
        ctx.rebalance(np.array([1.0, 0.0]))
      elif ctx.dp.i == 2:
        # Bar 2: rotate 100% from Stock A (@ 20.0) into Stock B (@ 10.0 -> 1000 shares = 10000)
        ctx.rebalance(np.array([0.0, 1.0]))

  acc = PortfolioAccount(10000.0, dp)
  runner = PortfolioRunner(acc)
  runner.run(RotateStrategy())

  # At bar 2 after rotation: Stock A sold (0 shares), Stock B bought (1000 shares @ 10.0)
  assert acc.positions[0].shares == 0.0
  assert acc.positions[1].shares == 1000.0
  # At bar 3: Stock B rises to 30.0 -> portfolio asset = 30000.0, ror = 2.0 (200%)
  assert np.isclose(acc.portfolio_assets[3], 30000.0)
  assert np.isclose(acc.portfolio_ror[3], 2.0)


def test_selector_strategy_integration():
  # 2 stocks, 5 bars:
  # Stock A rises first then stays flat; Stock B surges on bar 2+
  prices_a = np.array([10.0, 12.0, 12.0, 12.0, 12.0])
  prices_b = np.array([10.0, 10.0, 15.0, 20.0, 25.0])
  prices = np.concatenate([prices_a, prices_b])

  dp = DataProvider(
    {
      "ts": np.tile(np.arange(5), 2),
      "open": prices,
      "close": prices,
      "price": prices,
      "dividend": np.zeros(10),
      "transfer_shares": np.zeros(10),
      "right_shares": np.zeros(10),
      "right_price": np.zeros(10),
    },
    objs=["SH600000", "SZ000001"],
  )

  class MomentumTop1Selector(Selector):
    def step00(self, stocks: list[str]) -> list[str]:
      """Pick the stock with highest 1-bar return."""
      panel = self.dp.load_kline(stocks, lastN=2)
      ret = panel.last("close") / panel.offset("close", -2) - 1.0
      df = panel.cross_section(ret=ret).sort("ret", descending=True)
      self.factors = df
      return df["obj"].head(1).to_list()

  acc = PortfolioAccount(12000.0, dp)
  runner = PortfolioRunner(acc)
  strat = SelectorStrategy(MomentumTop1Selector(), rebalance_period=1, min_bars=1, max_holdings=1)
  runner.run(strat)

  # Bar 1: SH600000 had higher return (12/10 vs 10/10), buys SH600000 (1000 shares @ 12.0 = 12000)
  # Bar 2: SZ000001 had higher return (15/10 vs 12/12), rotates into SZ000001 (800 shares @ 15.0 = 12000)
  # Bar 4: SZ000001 reaches 25.0 -> 800 * 25.0 = 20000.0
  assert acc.positions[0].shares == 0.0
  assert acc.positions[1].shares == 800.0
  assert np.isclose(acc.portfolio_assets[4], 20000.0)


def test_portfolio_strategy_cli_and_report(tmp_path, monkeypatch):
  import sys
  from mbt.run.cli import parse_args
  from mbt.run import build_portfolio_report

  strat_file = tmp_path / "port_strat.py"
  strat_file.write_text(
    """
import mbt
import numpy as np

class MyPortfolioStrategy(mbt.PortfolioStrategy):
  def execute(self, ctx: mbt.Context):
    if ctx.dp.i == 1:
      # Equal weight across tradable stocks
      ctx.rebalance(np.array([0.5, 0.5]))
""",
    encoding="utf-8",
  )

  monkeypatch.setattr(
    sys,
    "argv",
    [
      "mbt.run.cli",
      str(strat_file),
      "-s",
      "SH600000",
      "SZ000001",
      "-B",
      "SH000300",
      "-c",
      "200000",
    ],
  )
  req, _, _, _ = parse_args()
  assert getattr(req, "portfolio", False) is True
  assert getattr(req, "cash", None) == 200000.0
  assert req.objs == ["SH000300", "SH600000", "SZ000001"]

  # Run portfolio strategy with tradable_start=1 (group 0 is benchmark SH000300)
  prices_bench = np.array([100.0, 101.0, 102.0, 103.0])
  prices_a = np.array([10.0, 10.0, 12.0, 15.0])
  prices_b = np.array([20.0, 20.0, 22.0, 24.0])
  prices = np.concatenate([prices_bench, prices_a, prices_b])

  dp = DataProvider(
    {
      "ts": np.tile(np.arange(4), 3),
      "open": prices,
      "close": prices,
      "price": prices,
      "bw_price": prices,
      "fw_price": prices,
      "dividend": np.zeros(12),
      "transfer_shares": np.zeros(12),
      "right_shares": np.zeros(12),
      "right_price": np.zeros(12),
    },
    objs=["SH000300", "SH600000", "SZ000001"],
  )
  acc = PortfolioAccount(200000.0, dp, tradable_start=1)
  runner = PortfolioRunner(acc)
  runner.run(req.strategy)

  # Group 0 (benchmark) must never be traded
  assert acc.positions[0].shares == 0.0
  assert acc.positions[1].shares == 10000.0
  assert acc.positions[2].shares == 5000.0

  reports = build_portfolio_report(dp, acc)
  assert len(reports) == 1
  assert reports[0].obj == "PORTFOLIO"
  assert np.isclose(reports[0].profit, (10000 * 15.0 + 5000 * 24.0) / 200000.0 - 1.0)

