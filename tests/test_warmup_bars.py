import numpy as np
import polars as pl
from mbt import DataProvider, PortfolioAccount, PortfolioRunner, Strategy, PortfolioContext
from mbt import Account, Runner
from mbt.run.report import build_portfolio_report, build_report


def test_portfolio_warmup_bars_suppresses_trading_and_slices_report():
  # 10 bars total: first 4 bars are warmup, 6 bars are actual backtest
  prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
  dp = DataProvider(
    {
      "ts": np.arange(10),
      "open": prices,
      "close": prices,
      "price": prices,
      "bw_price": prices,
      "dividend": np.zeros(10),
      "transfer_shares": np.zeros(10),
      "right_shares": np.zeros(10),
      "right_price": np.zeros(10),
    },
    objs=["SH600000"],
  )
  dp.warmup_bars = 4

  class AggressiveStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):
      # Try to rebalance on EVERY bar
      ctx.rebalance(np.array([1.0]))

  acc = PortfolioAccount(10000.0, dp)
  runner = PortfolioRunner(acc)
  runner.run(AggressiveStrategy())

  # Check that NO trades happened in bars 0, 1, 2, 3
  warmup_trades = [r for r in acc.records if r.bar < 4]
  assert len(warmup_trades) == 0

  # First trade happened at bar 4
  assert any(r.bar == 4 for r in acc.records)

  # Check report
  reports = build_portfolio_report(dp, acc)
  rep = reports[0]
  # Report dates must start at bar 4
  assert len(rep.date) == 6
  assert rep.date[0] == 4
  assert rep.date[-1] == 9
  # Initial cash at bar 4 was full 10000.0, so ror starts from 0 at bar 4
  assert rep.ror[0] == 0.0


def test_single_runner_warmup_bars():
  prices = np.array([10.0, 10.0, 10.0, 10.0, 12.0, 15.0])
  dp = DataProvider(
    {
      "ts": np.arange(6),
      "open": prices,
      "close": prices,
      "price": prices,
      "bw_price": prices,
      "dividend": np.zeros(6),
      "transfer_shares": np.zeros(6),
      "right_shares": np.zeros(6),
      "right_price": np.zeros(6),
    },
    objs=["SH600000"],
  )
  dp.warmup_bars = 3

  class BuyEveryBarStrategy(Strategy):
    def execute(self, ctx):
      ctx.buy(np.array([True]), percent=1.0)

  account = Account(10000.0, dp)
  runner = Runner(account)
  runner.run(BuyEveryBarStrategy())

  # Reports must be sliced to 6 - 3 = 3 bars
  reports = build_report(dp, runner.ctx.accounts)
  rep = reports[0]
  assert len(rep.date) == 3
  assert rep.date[0] == 3
  assert rep.date[-1] == 5
