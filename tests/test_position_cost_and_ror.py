from mbt import DataProvider, Position, Account, Operation
from mbt.portfolio import PortfolioAccount, PortfolioContext
from mbt.run.validator import check_strategy_source
import numpy as np
import pytest


def test_position_weighted_average_and_liquidation():
  # Buy 100 shares @ 10.0
  p1 = Position(100.0, 10.0)
  assert p1.shares == 100.0
  assert p1.raw_cost_price == 10.0
  assert p1.cost_basis == 1000.0
  assert p1.raw_mark_price == 10.0
  assert p1.holding_ror == 0.0
  assert p1.holding_pnl == 0.0

  # Buy 100 shares @ 20.0
  p2 = Position(100.0, 20.0)
  p_combined = p1 + p2
  assert p_combined.shares == 200.0
  # Moving weighted average cost: (1000 + 2000) / 200 = 15.0
  assert p_combined.raw_cost_price == 15.0
  assert p_combined.cost_basis == 3000.0

  # If mark price rises to 18.0
  p_combined.raw_mark_price = 18.0
  assert p_combined.market_value == 3600.0
  assert p_combined.holding_pnl == 600.0
  assert np.isclose(p_combined.holding_ror, (18.0 - 15.0) / 15.0)  # +20%

  # Sell 50 shares
  p_sold = Position(50.0, 18.0)
  p_remaining = p_combined - p_sold
  assert p_remaining.shares == 150.0
  # Partial liquidation preserves per-share average cost!
  assert p_remaining.raw_cost_price == 15.0
  assert p_remaining.cost_basis == 2250.0


def test_corporate_action_split_dilution():
  # Simulate 10送10 (transfer_shares = 1.0)
  dp = DataProvider(
    {
      "ts": np.arange(3),
      "price": np.array([20.0, 10.0, 11.0]),  # Unadjusted raw price: halves on bar 1 due to split
      "close": np.array([10.0, 10.0, 11.0]),  # Forward-adjusted price: already adjusted
      "dividend": np.zeros(3),
      "transfer_shares": np.array([0.0, 1.0, 0.0]),
      "right_shares": np.zeros(3),
      "right_price": np.zeros(3),
    },
    objs=["SZ000001"],
  )
  acc = PortfolioAccount(100000.0, dp)

  # Bar 0: Buy 1000 shares @ 20.0 (cost = 20,000)
  acc._buy_group(0, 0, 20000.0, 20.0)
  pos = acc.position("SZ000001")
  assert pos.shares == 1000.0
  assert pos.raw_cost_price == 20.0
  assert pos.cost_basis == 20000.0
  assert np.isclose(pos.adj_cost_price, 10.0)  # 20 * (10 / 20)

  # Bar 1: Corporate action (10送10) applies
  acc.apply_corporate_actions(1)
  pos = acc.position("SZ000001")
  assert pos.shares == 2000.0
  # Cost basis remains 20,000, per-share cost halves to 10.0!
  assert pos.raw_cost_price == 10.0
  assert pos.cost_basis == 20000.0
  assert np.isclose(pos.adj_cost_price, 5.0)

  # Update mark-to-market with post-split price 10.0
  acc.update_mark_to_market(1, np.array([10.0]))
  # Holding return MUST BE 0.0%, NOT -50%!
  assert np.isclose(acc.holding_ror("SZ000001"), 0.0)
  assert np.isclose(pos.holding_ror, 0.0)
  assert np.isclose(pos.holding_pnl, 0.0)


def test_corporate_action_cash_dividend():
  # Simulate cash dividend of 1.0 per share
  dp = DataProvider(
    {
      "ts": np.arange(2),
      "price": np.array([10.0, 9.0]),  # Unadjusted raw price: ex-dividend drops to 9.0
      "close": np.array([9.0, 9.0]),
      "dividend": np.array([0.0, 1.0]),
      "transfer_shares": np.zeros(2),
      "right_shares": np.zeros(2),
      "right_price": np.zeros(2),
    },
    objs=["SH600000"],
  )
  acc = PortfolioAccount(10000.0, dp)
  acc._buy_group(0, 0, 10000.0, 10.0)

  # Bar 1: Dividend distributed
  acc.apply_corporate_actions(1)
  pos = acc.position("SH600000")
  assert pos.dividends == 1000.0  # 1000 shares * 1.0

  acc.update_mark_to_market(1, np.array([9.0]))
  # Market value (9000) + dividends (1000) - cost (10000) == 0 -> return is 0%
  assert np.isclose(acc.holding_ror("SH600000"), 0.0)
  assert np.isclose(acc.holding_pnl("SH600000"), 0.0)


def test_portfolio_account_holding_inquiry_apis():
  dp = DataProvider(
    {
      "ts": np.arange(3),
      "price": np.array([10.0, 12.0, 15.0]),
      "close": np.array([10.0, 12.0, 15.0]),
      "dividend": np.zeros(3),
      "transfer_shares": np.zeros(3),
      "right_shares": np.zeros(3),
      "right_price": np.zeros(3),
    },
    objs=["SH600000"],
  )
  acc = PortfolioAccount(10000.0, dp)
  ctx = PortfolioContext(acc)

  # Initial check
  assert acc.holding_ror("SH600000") == 0.0
  assert acc.holding_days("SH600000") == 0
  assert ctx.holding_ror("SH600000") == 0.0

  # Buy at bar 0
  acc._buy_group(0, 0, 10000.0, 10.0)
  assert acc.holding_days("SH600000", current_bar=0) == 0

  # Bar 1 mark-to-market
  acc.update_mark_to_market(1, np.array([12.0]))
  assert np.isclose(acc.holding_ror("SH600000"), 0.20)
  assert np.isclose(ctx.holding_ror("SH600000"), 0.20)
  assert np.isclose(acc.holding_pnl("SH600000"), 2000.0)
  assert acc.cost_price("SH600000", adjusted=False) == 10.0
  assert acc.holding_days("SH600000", current_bar=1) == 1


def test_validator_detects_cross_dimension_anti_pattern():
  bad_code = '''
def execute(ctx):
  curr_price = ctx.close[0]
  cost_price = pos.price
  hold_ror = (curr_price - cost_price) / cost_price
'''
  warnings = check_strategy_source(bad_code, filename="bad_strat.py")
  assert len(warnings) > 0
  assert "Cross-dimensional price arithmetic detected" in warnings[0]

  # Test fail_fast mode raises ValueError
  with pytest.raises(ValueError, match="Cross-dimensional price arithmetic detected"):
    check_strategy_source(bad_code, filename="bad_strat.py", fail_fast=True)

  # Good code using framework holding_ror API
  good_code = '''
def execute(ctx):
  if ctx.holding_ror("SH600000") <= -0.08:
    ctx.sell("SH600000")
'''
  good_warnings = check_strategy_source(good_code, filename="good_strat.py")
  assert len(good_warnings) == 0
