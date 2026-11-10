import collections
import numpy as np
import pytest
from mbt import (
  DataProvider,
  Strategy,
  PortfolioStrategy,
  Account,
  PortfolioAccount,
  Runner,
  PortfolioRunner,
  Context,
  PortfolioContext,
)
from mbt.run.report import (
  build_report,
  build_portfolio_report,
  build_markdown_summary,
  build_json_report,
)


def create_sample_dp(bars: int = 5, objs: list[str] = None):
  if objs is None:
    objs = ["SH600000"]
  n_objs = len(objs)
  total = bars * n_objs
  prices = np.full(total, 10.0, dtype=np.float64)
  dates = np.tile(np.arange(bars, dtype=np.int64), n_objs)
  data = {
    "ts": dates,
    "open": prices,
    "close": prices,
    "price": prices,
    "bw_price": prices,
    "dividend": np.zeros(total),
    "transfer_shares": np.zeros(total),
    "right_shares": np.zeros(total),
    "right_price": np.zeros(total),
  }
  return DataProvider(data, objs=objs)


def test_single_runner_finalize():
  dp = create_sample_dp(5, ["SH600000"])

  class MyStrategy(Strategy):
    def __init__(self):
      super().__init__()
      self.called_finalize = False
      self.executed_bars = 0

    def execute(self, ctx: Context):
      self.executed_bars += 1

    def finalize(self, ctx: Context):
      self.called_finalize = True
      return {
        "executed_bars": self.executed_bars,
        "final_asset": float(ctx.accounts[0].asset),
      }

  strat = MyStrategy()
  acc = Account(10000.0, dp)
  runner = Runner(acc)
  runner.run(strat)

  assert strat.called_finalize is True
  assert strat.executed_bars == 4
  assert runner.diagnostics == {
    "executed_bars": 4,
    "final_asset": 10000.0,
  }

  # Check report integration
  reports = build_report(dp, runner.ctx.accounts, diagnostics=runner.diagnostics)
  assert reports[0].diagnostics == runner.diagnostics

  # Check JSON report
  json_rep = build_json_report(reports)
  assert "diagnostics" in json_rep["SH600000"]
  assert json_rep["SH600000"]["diagnostics"]["executed_bars"] == 4

  # Check Markdown summary
  md = build_markdown_summary(reports, account=acc)
  assert "## Strategy Diagnostics" in md
  assert "- **executed_bars**: 4" in md


def test_portfolio_runner_finalize_exit_reasons():
  dp = create_sample_dp(5, ["SH600000"])

  class ExitReasonStrategy(PortfolioStrategy):
    def prepare(self, ctx: PortfolioContext):
      self.exit_reasons = collections.Counter()

    def execute(self, ctx: PortfolioContext):
      # Simulate recording exit reasons
      if ctx.bar == 1:
        self.exit_reasons["discount_closed"] += 2
      elif ctx.bar == 2:
        self.exit_reasons["stop_loss"] += 1
      elif ctx.bar == 3:
        self.exit_reasons["leverage_up"] += 3

    def finalize(self, ctx: PortfolioContext):
      return {
        "Exit Reasons": dict(self.exit_reasons),
        "Total Exits": sum(self.exit_reasons.values()),
      }

  strat = ExitReasonStrategy()
  acc = PortfolioAccount(100000.0, dp)
  runner = PortfolioRunner(acc)
  runner.run(strat)

  assert runner.diagnostics == {
    "Exit Reasons": {
      "discount_closed": 2,
      "stop_loss": 1,
      "leverage_up": 3,
    },
    "Total Exits": 6,
  }
  assert acc.diagnostics == runner.diagnostics

  # Check report integration
  reports = build_portfolio_report(dp, acc, diagnostics=runner.diagnostics)
  assert reports[0].diagnostics == runner.diagnostics

  # Check JSON export
  json_rep = build_json_report(reports)
  assert "diagnostics" in json_rep["PORTFOLIO"]
  assert json_rep["PORTFOLIO"]["diagnostics"]["Total Exits"] == 6

  # Check Markdown summary rendering
  md = build_markdown_summary(reports, account=acc)
  assert "## Strategy Diagnostics" in md
  assert "### Exit Reasons" in md
  assert "| stop_loss | 1 |" in md
  assert "| discount_closed | 2 |" in md
  assert "| leverage_up | 3 |" in md
  assert "- **Total Exits**: 6" in md


def test_finalize_default_noop():
  dp = create_sample_dp(3, ["SH600000"])

  class NoOpStrategy(Strategy):
    def execute(self, ctx: Context):
      pass

  strat = NoOpStrategy()
  acc = Account(10000.0, dp)
  runner = Runner(acc)
  runner.run(strat)

  assert runner.diagnostics is None
  reports = build_report(dp, runner.ctx.accounts)
  assert reports[0].diagnostics is None
