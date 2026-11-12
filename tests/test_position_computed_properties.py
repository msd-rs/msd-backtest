import numpy as np
import pytest
from mbt import Position, Account, DataProvider, PortfolioAccount, PortfolioRunner, Strategy, Runner


def test_position_standalone_computed_properties():
  # Zero position
  pos_zero = Position.zero()
  assert pos_zero.shares == 0.0
  assert pos_zero.ror == 0.0
  assert pos_zero.unrealized_pnl == 0.0
  assert pos_zero.pnl == 0.0
  assert pos_zero.holding_days == 0
  assert not pos_zero.is_holding

  # Position with positive unrealized profit (+25%)
  pos = Position(
    shares=1000.0,
    raw_cost_price=10.0,
    raw_mark_price=12.0,
    dividends=500.0,
    adj_cost_price=9.5,
    holding_days=5,
  )
  assert pos.shares == 1000.0
  assert pos.cost_basis == 10000.0
  assert pos.market_value == 12000.0
  assert pos.unrealized_pnl == (12000.0 + 500.0) - 10000.0  # 2500.0
  assert pos.pnl == 2500.0
  assert pos.holding_pnl == 2500.0
  assert pos.ror == 2500.0 / 10000.0  # +25%
  assert pos.holding_ror == 0.25
  assert pos.holding_days == 5
  assert pos.holding_bars == 5
  assert pos.is_holding


def test_single_account_position_lifecycle():
  """验证单标的账户交易流程中 Position 对象的 ror, unrealized_pnl 与 holding_days 联动."""
  n_bars = 10
  dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-11"), dtype="datetime64[D]")
  prices = np.array([10.0, 10.5, 11.0, 11.5, 9.0, 9.5, 10.0, 10.0, 10.0, 10.0])

  dp = DataProvider(
    {
      "ts": dates,
      "open": prices,
      "high": prices * 1.01,
      "low": prices * 0.99,
      "close": prices,
      "price": prices,
      "dividend": np.zeros(n_bars),
      "transfer_shares": np.zeros(n_bars),
      "right_shares": np.zeros(n_bars),
      "right_price": np.zeros(n_bars),
    },
    objs=["SH600000"],
  )

  account = Account(100000.0, dp)

  class SingleLifecycleStrategy(Strategy):
    def execute(self, ctx):
      pos = ctx.position()
      if ctx.bar == 1:
        # Buy on bar 1 (@ 10.5)
        ctx.buy(True, percent=1.0)
      elif ctx.bar == 2:
        # Bar 2 (@ 11.0), 1 day held
        assert pos.is_holding
        assert pos.holding_days == 1
        assert np.isclose(pos.raw_cost_price, 10.5)
        assert np.isclose(pos.raw_mark_price, 11.0)
        assert np.isclose(pos.ror, (11.0 - 10.5) / 10.5)
        assert np.isclose(pos.unrealized_pnl, pos.pnl)
      elif ctx.bar == 4:
        # Bar 4 (@ 9.0), 3 days held
        assert pos.holding_days == 3
        # Price dropped to 9.0 -> loss
        assert pos.ror < 0.0
        assert pos.unrealized_pnl < 0.0
        # Liquidate
        ctx.sell(True, percent=1.0)
      elif ctx.bar == 5:
        # Flat position
        assert not pos.is_holding
        assert pos.holding_days == 0
        assert pos.ror == 0.0
        assert pos.unrealized_pnl == 0.0

  runner = Runner(account)
  runner.run(SingleLifecycleStrategy())


def test_portfolio_account_position_lifecycle():
  """验证多标的组合账户中通过 ctx.position(obj) 获取的各计算属性."""
  n_bars = 6
  dates = np.tile(np.arange(6), 2)
  p_a = np.array([10.0, 10.0, 12.0, 13.0, 14.0, 15.0])
  p_b = np.array([20.0, 20.0, 18.0, 17.0, 16.0, 15.0])

  dp = DataProvider(
    {
      "ts": dates,
      "open": np.concatenate([p_a, p_b]),
      "close": np.concatenate([p_a, p_b]),
      "price": np.concatenate([p_a, p_b]),
      "raw_close": np.concatenate([p_a, p_b]),
      "dividend": np.zeros(12),
      "transfer_shares": np.zeros(12),
      "right_shares": np.zeros(12),
      "right_price": np.zeros(12),
    },
    objs=["SH600000", "SZ000001"],
  )

  account = PortfolioAccount(100000.0, dp)

  class PortfolioLifecycleStrategy(Strategy):
    portfolio: bool = True

    def execute(self, ctx):
      pos_a = ctx.position("SH600000")
      pos_b = ctx.position("SZ000001")

      if ctx.bar == 1:
        # Buy both on bar 1 (@ 10.0 and 20.0)
        ctx.buy("SH600000", percent=0.5)
        ctx.buy("SZ000001", percent=0.5)
      elif ctx.bar == 2:
        # 1 bar held (bar 2: p_a=12.0 -> +20%, p_b=18.0 -> -10%)
        assert pos_a.is_holding
        assert pos_a.holding_days == 1
        assert np.isclose(pos_a.ror, 0.20)
        assert pos_a.unrealized_pnl > 0.0

        assert pos_b.is_holding
        assert pos_b.holding_days == 1
        assert np.isclose(pos_b.ror, -0.10)
        assert pos_b.unrealized_pnl < 0.0
      elif ctx.bar == 3:
        assert pos_a.holding_days == 2
        # Sell A on bar 3
        ctx.sell("SH600000", percent=1.0)
      elif ctx.bar == 4:
        # A is sold
        assert not pos_a.is_holding
        assert pos_a.holding_days == 0
        assert pos_a.ror == 0.0
        # B is still held (3 days held)
        assert pos_b.is_holding
        assert pos_b.holding_days == 3

  runner = PortfolioRunner(account)
  runner.run(PortfolioLifecycleStrategy())
