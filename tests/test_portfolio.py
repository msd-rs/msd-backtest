import numpy as np
import polars as pl
from mbt import (
  DataProvider,
  PortfolioAccount,
  PortfolioContext,
  PortfolioRecord,
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


def test_portfolio_records_and_trades_df():
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
        ctx.rebalance(np.array([1.0, 0.0]))
      elif ctx.dp.i == 2:
        ctx.rebalance(np.array([0.0, 1.0]))

  acc = PortfolioAccount(10000.0, dp)
  # Empty before run
  assert len(acc.records) == 0
  empty_df = acc.trades_df()
  assert empty_df.shape == (0, 12)
  assert "obj" in empty_df.columns
  assert "action" in empty_df.columns

  runner = PortfolioRunner(acc)
  runner.run(RotateStrategy())

  assert len(acc.records) == 3

  # Trade 1: Buy SH600000 at bar 1
  r0 = acc.records[0]
  assert isinstance(r0, PortfolioRecord)
  assert r0.bar == 1
  assert r0.obj == "SH600000"
  assert r0.action == "BUY"
  assert r0.shares == 500.0
  assert r0.deal_price == 20.0
  assert r0.amount == 10000.0
  assert r0["obj"] == "SH600000"

  # Trade 2: Sell SH600000 at bar 2
  r1 = acc.records[1]
  assert r1.bar == 2
  assert r1.obj == "SH600000"
  assert r1.action == "SELL"
  assert r1.shares == 500.0
  assert r1.deal_price == 20.0
  assert r1.amount == 10000.0

  # Trade 3: Buy SZ000001 at bar 2
  r2 = acc.records[2]
  assert r2.bar == 2
  assert r2.obj == "SZ000001"
  assert r2.action == "BUY"
  assert r2.shares == 1000.0
  assert r2.deal_price == 10.0
  assert r2.amount == 10000.0

  df = acc.trades_df()
  assert df.shape == (3, 12)
  assert df["action"].to_list() == ["BUY", "SELL", "BUY"]
  assert df["obj"].to_list() == ["SH600000", "SH600000", "SZ000001"]
  assert df["shares"].to_list() == [500.0, 500.0, 1000.0]

  # Test alias records_df
  df_alias = acc.records_df()
  assert df_alias.equals(df)


def test_portfolio_dynamic_universe_and_runoff_mode():
  prices_a = np.array([10.0, 10.0, 12.0, 14.0])
  prices_b = np.array([20.0, 20.0, 25.0, 30.0])
  prices_c = np.array([5.0, 5.0, 5.0, 5.0])
  prices = np.concatenate([prices_a, prices_b, prices_c])

  dp = DataProvider(
    {
      "ts": np.tile(np.arange(4), 3),
      "open": prices,
      "close": prices,
      "price": prices,
      "dividend": np.zeros(12),
      "transfer_shares": np.zeros(12),
      "right_shares": np.zeros(12),
      "right_price": np.zeros(12),
    },
    objs=["SH600000", "SZ000001", "SZ000002"],
  )

  class DynamicUniverseStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):
      if ctx.bar == 1:
        # Bar 1: whitelist contains only A and B
        ctx.set_universe(["SH600000", "SZ000001"])
        assert ctx.universe == {"SH600000", "SZ000001"}

        # Attempt to buy C (not in universe) -> should be blocked by guard
        ctx.buy("SZ000002", percent=0.2)
        # Buy A (in universe)
        ctx.buy("SH600000", percent=0.5)

      elif ctx.bar == 2:
        # After Bar 1 execution: A was bought, C was blocked
        assert ctx.account.has_position("SH600000")
        assert not ctx.account.has_position("SZ000002")
        assert "SH600000" in ctx.account.held_objs

        # Bar 2: Universe rotates to only B and C (A is dropped -> becomes Runoff!)
        ctx.set_universe(["SZ000001", "SZ000002"])
        assert "SH600000" in ctx.account.runoff_objs(ctx.universe)

        # Attempt to buy A again -> blocked because it's no longer in universe
        ctx.buy("SH600000", percent=0.5)
        # But buy B (now in universe)
        ctx.buy("SZ000001", percent=0.5)

      elif ctx.bar == 3:
        # Both A and B are held
        assert ctx.account.has_position("SH600000")
        assert ctx.account.has_position("SZ000001")
        assert "SH600000" in ctx.account.runoff_objs(ctx.universe)

        # Natural liquidation (Runoff mode): sell A even though it is not in current universe!
        ctx.sell("SH600000", percent=1.0)

  acc = PortfolioAccount(20000.0, dp)
  runner = PortfolioRunner(acc)
  runner.run(DynamicUniverseStrategy())

  # At end of backtest (Bar 3 execution finished):
  # A was sold (0 shares), B is still held
  assert not acc.has_position("SH600000")
  assert acc.has_position("SZ000001")
  assert not acc.has_position("SZ000002")
  assert acc.runoff_objs(runner.ctx.universe) == []
  assert acc.held_objs == ["SZ000001"]


def test_portfolio_context_run_selector():
  prices_a = np.array([10.0, 12.0, 15.0, 20.0])
  prices_b = np.array([10.0, 10.0, 10.0, 10.0])
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

  class TopMoverSelector(Selector):
    def step00(self, stocks: list[str]) -> list[str]:
      panel = self.dp.load_kline(stocks, lastN=2)
      ret = panel.last("close") / panel.offset("close", -2) - 1.0
      df = panel.cross_section(ret=ret).sort("ret", descending=True)
      return df["obj"].head(1).to_list()

  class AdaptiveStrategy(Strategy):
    def __init__(self):
      super().__init__()
      self.selector = TopMoverSelector()

    def execute(self, ctx: PortfolioContext):
      if ctx.bar == 1:
        picked = ctx.run_selector(self.selector, lastN=2)
        assert picked == ["SH600000"]
        ctx.set_universe(picked)
        for s in ctx.universe:
          ctx.buy(s, percent=1.0)

  acc = PortfolioAccount(10000.0, dp)
  runner = PortfolioRunner(acc)
  runner.run(AdaptiveStrategy())

  assert acc.has_position("SH600000")
  assert not acc.has_position("SZ000001")
  assert acc.positions[0].shares == 800.0  # 10000 / 12.0 = 833.3 -> 800 shares @ 12.0


def test_portfolio_rebalance_tolerance_band():
  # Test that rebalance with tolerance suppresses micro trades but executes liquidations
  prices_a = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
  prices_b = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
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

  class TolStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):
      if ctx.bar == 1:
        # Initial 50/50 allocation
        ctx.rebalance(np.array([0.5, 0.5]))
      elif ctx.bar == 2:
        # Micro deviation within 5% tolerance: target 0.52 vs current ~0.50
        ctx.rebalance(np.array([0.52, 0.48]), tolerance=0.05)
      elif ctx.bar == 3:
        # Large deviation outside 5% tolerance: target 0.70 vs current ~0.50
        ctx.rebalance(np.array([0.70, 0.30]), tolerance=0.05)
      elif ctx.bar == 4:
        # Full liquidation on SZ000001 even within tolerance
        ctx.rebalance(np.array([1.0, 0.0]), tolerance=0.05)

  acc = PortfolioAccount(10000.0, dp)
  runner = PortfolioRunner(acc)
  runner.run(TolStrategy())

  records = acc.records
  # At bar 1: 2 buys (SH600000, SZ000001)
  assert sum(1 for r in records if r.bar == 1) == 2
  # At bar 2: deviation <= 0.05, so NO trades at bar 2!
  assert sum(1 for r in records if r.bar == 2) == 0
  # At bar 3: deviation > 0.05, trades occur
  assert sum(1 for r in records if r.bar == 3) > 0
  # At bar 4: SZ000001 must be completely liquidated
  assert acc.positions[1].shares == 0.0


def test_portfolio_context_panel_warning_and_slice(caplog):
  import logging
  prices = np.ones(10)
  dp = DataProvider(
    {
      "ts": np.arange(10),
      "open": prices,
      "close": prices,
      "price": prices,
      "dividend": np.zeros(10),
      "transfer_shares": np.zeros(10),
      "right_shares": np.zeros(10),
      "right_price": np.zeros(10),
    },
    objs=["SH600000"],
  )

  class InspectStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):
      if ctx.bar == 2:
        # Calling ctx.panel() without allow_future should log a warning
        p_full = ctx.panel()
        assert p_full.bars == 10
        # Calling ctx.panel(sliced=True) should return sliced panel
        p_sliced = ctx.panel(sliced=True)
        assert p_sliced.bars == 3  # bar 0, 1, 2
        # Calling ctx.slice_panel(lastN=2)
        p_slice2 = ctx.slice_panel(lastN=2)
        assert p_slice2.bars == 2

  acc = PortfolioAccount(10000.0, dp)
  runner = PortfolioRunner(acc)
  with caplog.at_level(logging.WARNING):
    runner.run(InspectStrategy())

  assert any("Calling ctx.panel() during execute()" in rec.message for rec in caplog.records)


def test_historical_panel_selector_fin_only_year_warning(caplog):
  import logging
  from mbt.portfolio import _HistoricalPanelSelectorDataProvider
  from mbt.select.panel import Panel

  df = pl.DataFrame({
    "close": [10.0, 11.0],
    "f001": [1.0, 2.0],
  })
  panel = Panel(df, ["SH600000"], 2)

  # When backtest was NOT loaded with fin_only_year (fin_only_year=False),
  # requesting fin_only_year=True with fin_fields triggers a warning
  dp_slice = _HistoricalPanelSelectorDataProvider(panel, fin_only_year=False)
  with caplog.at_level(logging.WARNING):
    dp_slice.load_snapshot(["SH600000"], fin_fields=["f001"], fin_only_year=True)
  assert any("requested fin_only_year=True with fin_fields" in rec.message for rec in caplog.records)

  # When backtest WAS loaded with fin_only_year=True, no warning
  caplog.clear()
  dp_slice_ok = _HistoricalPanelSelectorDataProvider(panel, fin_only_year=True)
  with caplog.at_level(logging.WARNING):
    dp_slice_ok.load_snapshot(["SH600000"], fin_fields=["f001"], fin_only_year=True)
  assert not any("requested fin_only_year=True with fin_fields" in rec.message for rec in caplog.records)


def test_rebalance_and_buy_universe_whitelist_guard():
  # 2 stocks, 4 bars
  prices = np.full(8, 10.0)
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

  class WhitelistStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):
      if ctx.bar == 1:
        # Whitelist ONLY SH600000
        ctx.set_universe(["SH600000"])
        # Attempt to rebalance 50% into SH600000 and 50% into SZ000001
        ctx.rebalance(np.array([0.5, 0.5]))
      elif ctx.bar == 2:
        # In bar 2, verify SH600000 was bought, SZ000001 was NOT bought
        assert ctx.account.positions[0].shares > 0
        assert ctx.account.positions[1].shares == 0.0

        # Now whitelist ONLY SZ000001; SH600000 drops into Runoff mode
        ctx.set_universe(["SZ000001"])
        # Attempt to allocate 80% to SH600000 (outside universe) and 20% to SZ000001
        ctx.rebalance(np.array([0.8, 0.2]))

  acc = PortfolioAccount(10000.0, dp)
  runner = PortfolioRunner(acc)
  runner.run(WhitelistStrategy())

  # Bar 1 result checked in bar 2.
  # At the end of bar 2:
  # SH600000 was outside universe in bar 2, so its 80% request was capped at current holding (cannot buy more!)
  # SZ000001 was inside universe in bar 2, so it was bought!
  assert acc.positions[1].shares > 0.0


def test_set_universe_and_run_selector_missing_objs_warning(caplog):
  import logging
  prices = np.full(8, 10.0)
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

  class MissingObjStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):
      if ctx.bar == 1:
        # Pass an obj not in preloaded pool
        ctx.set_universe(["SH600000", "SH999999"])

  acc = PortfolioAccount(10000.0, dp)
  runner = PortfolioRunner(acc)
  with caplog.at_level(logging.WARNING):
    runner.run(MissingObjStrategy())

  assert any("set_universe: 1 obj(s) not in pre-loaded pool" in rec.message for rec in caplog.records)



