from mbt import Account, Context, DataProvider, Fee, Operation, PortfolioStrategy, Position, Runner, Strategy
from mbt.portfolio import PortfolioAccount, PortfolioContext, PortfolioRunner
from mbt.run.validator import check_strategy_source
from mbt.shared import apply_price_adjustment
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
      "close": np.array([20.0, 20.0, 22.0]),  # Geometric backward-adjusted price: continuous across split
      "dividend": np.zeros(3),
      "transfer_shares": np.array([0.0, 1.0, 0.0]),
      "right_shares": np.zeros(3),
      "right_price": np.zeros(3),
    },
    objs=["SZ000001"],
  )
  acc = PortfolioAccount(100000.0, dp, fee=Fee.zero())

  # Bar 0: Buy 1000 shares @ 20.0 (cost = 20,000)
  acc._buy_group(0, 0, 20000.0, 20.0)
  pos = acc.position("SZ000001")
  assert pos.shares == 1000.0
  assert pos.raw_cost_price == 20.0
  assert pos.cost_basis == 20000.0
  assert np.isclose(pos.adj_cost_price, 20.0)  # 20 * (20 / 20)

  # Bar 1: Corporate action (10送10) applies
  acc.apply_corporate_actions(1)
  pos = acc.position("SZ000001")
  assert pos.shares == 2000.0
  # Cost basis remains 20,000, raw per-share cost halves to 10.0, while adj_cost_price stays 20.0
  assert pos.raw_cost_price == 10.0
  assert pos.cost_basis == 20000.0
  assert np.isclose(pos.adj_cost_price, 20.0)

  # Update mark-to-market with post-split price 10.0
  acc.update_mark_to_market(1, np.array([10.0]))
  # Holding return MUST BE 0.0%, NOT -50%, and close / adj_cost_price - 1 matches 0.0%!
  assert np.isclose(acc.holding_ror("SZ000001"), 0.0)
  assert np.isclose(pos.holding_ror, 0.0)
  assert np.isclose(pos.holding_pnl, 0.0)
  assert np.isclose(dp.all("close")[1] / pos.adj_cost_price - 1.0, 0.0)

  # Bar 2: stock rises +10% (raw 11.0, adjusted 22.0)
  acc.update_mark_to_market(2, np.array([11.0]))
  assert np.isclose(acc.holding_ror("SZ000001"), 0.10)
  assert np.isclose(dp.all("close")[2] / pos.adj_cost_price - 1.0, 0.10)


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
  acc = PortfolioAccount(10000.0, dp, fee=Fee.zero())
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
  acc = PortfolioAccount(10000.0, dp, fee=Fee.zero())
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


def _build_adjusted_dp(
  raw_close: np.ndarray,
  dividend: np.ndarray,
  transfer_shares: np.ndarray,
  right_shares: np.ndarray,
  right_price: np.ndarray,
  obj: str = "SH600519",
) -> DataProvider:
  n = len(raw_close)
  data = {
    "ts": np.arange(n, dtype=np.int64) * 86400,
    "open": raw_close.copy(),
    "high": raw_close.copy(),
    "low": raw_close.copy(),
    "close": raw_close.copy(),
    "dividend": dividend.astype(np.float64).copy(),
    "transfer_shares": transfer_shares.astype(np.float64).copy(),
    "right_shares": right_shares.astype(np.float64).copy(),
    "right_price": right_price.astype(np.float64).copy(),
  }
  apply_price_adjustment(data, groups=1)
  return DataProvider(data, objs=[obj])


def _keep_and_mark(acc: Account, bar: int, raw_price: float) -> None:
  acc.do_operation(bar, Operation.keep(raw_price))
  if acc.position.shares > 0:
    acc.position.raw_mark_price = float(raw_price)


def test_adj_cost_price_sh600519_split_and_dividend_all_engines():
  # Reproduces SH600519 crossing ex-date (div=0.5, tr=0.2):
  # Bar 0: 50.0, Bar 1: buy @ 51.90, Bar 2: 55.51 (+6.96%), Bar 3: ex-date 46.65 (+8.82%)
  raw_close = np.array([50.0, 51.90, 55.51, 46.65], dtype=np.float64)
  div = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float64)
  trans = np.array([0.0, 0.0, 0.0, 0.2], dtype=np.float64)
  zeros = np.zeros(4, dtype=np.float64)

  dp_base = _build_adjusted_dp(raw_close, div, trans, zeros, zeros, obj="SH600519")
  c2 = float(dp_base.all("close")[2])
  c3 = float(dp_base.all("close")[3])
  assert np.isclose(c2, 55.51)
  assert np.isclose(c3, 55.51 * (46.65 / ((55.51 - 0.5) / 1.2)))
  expected_ror2 = 55.51 / 51.90 - 1.0
  expected_ror3 = (1.2 * 46.65 + 0.5 - 51.90) / 51.90

  # 1. Python Account
  dp_py_acc = _build_adjusted_dp(raw_close, div, trans, zeros, zeros, obj="SH600519")
  acc = Account(100000.0, dp_py_acc, fee=Fee.zero())
  _keep_and_mark(acc, 0, float(raw_close[0]))
  acc.do_operation(1, Operation.buy(Position(1000.0, float(raw_close[1]))))
  _keep_and_mark(acc, 2, float(raw_close[2]))
  assert np.isclose(acc.position.adj_cost_price, 51.90)
  assert np.isclose(acc.holding_ror, expected_ror2)
  _keep_and_mark(acc, 3, float(raw_close[3]))
  assert np.isclose(acc.position.adj_cost_price, 51.90)
  assert np.isclose(acc.holding_ror, expected_ror3)
  assert np.isclose(c3 / acc.position.adj_cost_price - 1.0, c3 / 51.90 - 1.0)

  # 2. Python PortfolioAccount
  dp_py_port = _build_adjusted_dp(raw_close, div, trans, zeros, zeros, obj="SH600519")
  pacc = PortfolioAccount(100000.0, dp_py_port, fee=Fee.zero())
  pacc._buy_group(0, 1, 51900.0, float(raw_close[1]))
  pacc.update_mark_to_market(1, np.array([raw_close[1]]))
  pacc.update_mark_to_market(2, np.array([raw_close[2]]))
  pacc.apply_corporate_actions(3)
  pacc.update_mark_to_market(3, np.array([raw_close[3]]))
  ppos = pacc.position("SH600519")
  assert np.isclose(ppos.adj_cost_price, 51.90)
  assert np.isclose(ppos.holding_ror, expected_ror3)

  # 3. Rust Runner (execute + signals fast-path)
  observed_runner_adj: list[float] = []
  observed_runner_ror: list[float] = []

  class InspectStrat(Strategy):
    def execute(self, ctx: Context):
      if ctx.bar == 1:
        ctx.buy(np.array([True]), 1.0)
      elif ctx.bar >= 2:
        observed_runner_adj.append(ctx.cost_price(0, adjusted=True))
        observed_runner_ror.append(ctx.holding_ror(0))

  dp_rs_acc = _build_adjusted_dp(raw_close, div, trans, zeros, zeros, obj="SH600519")
  r_acc = Account(100000.0, dp_rs_acc, fee=Fee.zero())
  runner_exec = Runner(r_acc, exec_mode="close")
  runner_exec.run(InspectStrat())
  assert np.allclose(observed_runner_adj, [51.90, 51.90])
  assert np.allclose(observed_runner_ror, [expected_ror2, expected_ror3])
  assert np.isclose(runner_exec.ctx.accounts[0].position.adj_cost_price, 51.90)

  class FastStrat(Strategy):
    def signals(self, ctx: Context):
      buy = np.array([False, True, False, False])
      sell = np.zeros(4, dtype=bool)
      return buy, sell

  dp_rs_fast = _build_adjusted_dp(raw_close, div, trans, zeros, zeros, obj="SH600519")
  f_acc = Account(100000.0, dp_rs_fast, fee=Fee.zero())
  runner_fast = Runner(f_acc, exec_mode="close")
  runner_fast.run(FastStrat())
  assert np.isclose(runner_fast.ctx.accounts[0].position.adj_cost_price, 51.90)
  assert np.isclose(runner_fast.ctx.accounts[0].holding_ror, expected_ror3)

  # 4. Rust PortfolioRunner
  observed_port_adj: list[float] = []
  observed_port_ror: list[float] = []

  class InspectPortStrat(PortfolioStrategy):
    def execute(self, ctx: PortfolioContext):  # type: ignore[override]
      if ctx.bar == 1:
        ctx.buy("SH600519", 0.6)
      elif ctx.bar >= 2:
        observed_port_adj.append(ctx.cost_price("SH600519", adjusted=True))
        observed_port_ror.append(ctx.holding_ror("SH600519"))

  dp_rs_port = _build_adjusted_dp(raw_close, div, trans, zeros, zeros, obj="SH600519")
  rp_acc = PortfolioAccount(100000.0, dp_rs_port, fee=Fee.zero())
  PortfolioRunner(rp_acc, exec_mode="close").run(InspectPortStrat())
  assert np.allclose(observed_port_adj, [51.90, 51.90])
  assert np.allclose(observed_port_ror, [expected_ror2, expected_ror3])
  assert np.isclose(rp_acc.position("SH600519").adj_cost_price, 51.90)


def test_adj_cost_price_post_split_addon_buy_all_engines():
  # Bar 0: 20.0, Bar 1: buy 1000 @ 20.0, Bar 2: 10送10 (raw=10.0, close=20.0),
  # Bar 3: buy 1000 @ 11.0 (close=22.0), Bar 4: 12.0 (close=24.0)
  raw_close = np.array([20.0, 20.0, 10.0, 11.0, 12.0], dtype=np.float64)
  zeros = np.zeros(5, dtype=np.float64)
  trans = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)

  expected_adj = (2000.0 * 20.0 + 1000.0 * 22.0) / 3000.0
  expected_ror4 = (3000.0 * 12.0 - 31000.0) / 31000.0

  # 1. Python Account
  dp_py = _build_adjusted_dp(raw_close, zeros, trans, zeros, zeros, obj="SZ000001")
  acc = Account(100000.0, dp_py, fee=Fee.zero())
  _keep_and_mark(acc, 0, float(raw_close[0]))
  acc.do_operation(1, Operation.buy(Position(1000.0, float(raw_close[1]))))
  _keep_and_mark(acc, 2, float(raw_close[2]))
  assert np.isclose(acc.position.adj_cost_price, 20.0)
  acc.do_operation(3, Operation.buy(Position(1000.0, float(raw_close[3]))))
  assert np.isclose(acc.position.adj_cost_price, expected_adj)
  _keep_and_mark(acc, 4, float(raw_close[4]))
  assert np.isclose(acc.holding_ror, expected_ror4)
  assert np.isclose(dp_py.all("close")[4] / acc.position.adj_cost_price - 1.0, expected_ror4)

  # 2. Python PortfolioAccount
  dp_p = _build_adjusted_dp(raw_close, zeros, trans, zeros, zeros, obj="SZ000001")
  pacc = PortfolioAccount(100000.0, dp_p, fee=Fee.zero())
  pacc._buy_group(0, 1, 20000.0, float(raw_close[1]))
  pacc.update_mark_to_market(1, np.array([raw_close[1]]))
  pacc.apply_corporate_actions(2)
  pacc.update_mark_to_market(2, np.array([raw_close[2]]))
  assert np.isclose(pacc.position("SZ000001").adj_cost_price, 20.0)
  pacc._buy_group(0, 3, 11000.0, float(raw_close[3]))
  pacc.update_mark_to_market(3, np.array([raw_close[3]]))
  pacc.update_mark_to_market(4, np.array([raw_close[4]]))
  assert np.isclose(pacc.position("SZ000001").adj_cost_price, expected_adj)
  assert np.isclose(pacc.holding_ror("SZ000001"), expected_ror4)

  # 3. Rust Runner (execute)
  class AddonStrat(Strategy):
    def execute(self, ctx: Context):
      if ctx.bar == 1:
        ctx.buy(np.array([True]), 0.2)
      elif ctx.bar == 3:
        ctx.buy(np.array([True]), 11000.0 / 80000.0)

  dp_r = _build_adjusted_dp(raw_close, zeros, trans, zeros, zeros, obj="SZ000001")
  racc = Account(100000.0, dp_r, fee=Fee.zero())
  runner_addon = Runner(racc, exec_mode="close")
  runner_addon.run(AddonStrat())
  assert runner_addon.ctx.accounts[0].position.shares == 3000.0
  assert np.isclose(runner_addon.ctx.accounts[0].position.adj_cost_price, expected_adj)
  assert np.isclose(runner_addon.ctx.accounts[0].holding_ror, expected_ror4)

  # 4. Rust PortfolioRunner
  class AddonPortStrat(PortfolioStrategy):
    def execute(self, ctx: PortfolioContext):  # type: ignore[override]
      if ctx.bar == 1:
        ctx.buy("SZ000001", 0.2)
      elif ctx.bar == 3:
        ctx.buy("SZ000001", 11000.0 / 80000.0)

  dp_rp = _build_adjusted_dp(raw_close, zeros, trans, zeros, zeros, obj="SZ000001")
  rpacc = PortfolioAccount(100000.0, dp_rp, fee=Fee.zero())
  PortfolioRunner(rpacc, exec_mode="close").run(AddonPortStrat())
  assert rpacc.position("SZ000001").shares == 3000.0
  assert np.isclose(rpacc.position("SZ000001").adj_cost_price, expected_adj)
  assert np.isclose(rpacc.holding_ror("SZ000001"), expected_ror4)


def test_adj_cost_price_free_and_paid_rights_all_engines():
  # Subtest A: Free allotment (right_shares=0.2, right_price=0.0)
  raw_free = np.array([10.0, 10.0, 10.0 / 1.2, 11.0 / 1.2], dtype=np.float64)
  zeros4 = np.zeros(4, dtype=np.float64)
  rs_free = np.array([0.0, 0.0, 0.2, 0.0], dtype=np.float64)

  dp_free_py = _build_adjusted_dp(raw_free, zeros4, zeros4, rs_free, zeros4, obj="SH600000")
  acc_f = Account(20000.0, dp_free_py, fee=Fee.zero())
  _keep_and_mark(acc_f, 0, float(raw_free[0]))
  acc_f.do_operation(1, Operation.buy(Position(1000.0, float(raw_free[1]))))
  _keep_and_mark(acc_f, 2, float(raw_free[2]))
  _keep_and_mark(acc_f, 3, float(raw_free[3]))
  assert np.isclose(acc_f.position.adj_cost_price, 10.0)
  assert np.isclose(acc_f.holding_ror, 0.10)
  assert np.isclose(dp_free_py.all("close")[3] / acc_f.position.adj_cost_price - 1.0, 0.10)

  dp_free_p = _build_adjusted_dp(raw_free, zeros4, zeros4, rs_free, zeros4, obj="SH600000")
  pacc_f = PortfolioAccount(20000.0, dp_free_p, fee=Fee.zero())
  pacc_f._buy_group(0, 1, 10000.0, float(raw_free[1]))
  pacc_f.apply_corporate_actions(2)
  pacc_f.update_mark_to_market(3, np.array([raw_free[3]]))
  assert np.isclose(pacc_f.position("SH600000").adj_cost_price, 10.0)
  assert np.isclose(pacc_f.holding_ror("SH600000"), 0.10)

  # Subtest B: Paid rights issue (right_shares=0.2, right_price=5.0)
  # Buy 1000 @ 8.0 on bar 1, bar 2 rises to 10.0, bar 3 paid rights @ 5.0 (ex_ref = 11/1.2),
  # bar 4 rises +10% (close = 11.0).
  raw_paid = np.array([8.0, 8.0, 10.0, 11.0 / 1.2, 1.1 * (11.0 / 1.2)], dtype=np.float64)
  zeros5 = np.zeros(5, dtype=np.float64)
  rs_paid = np.array([0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float64)
  rp_paid = np.array([0.0, 0.0, 0.0, 5.0, 0.0], dtype=np.float64)
  expected_adj_paid = 90.0 / 11.0
  expected_ror3 = 2.0 / 9.0
  expected_ror4 = 31.0 / 90.0

  # 1. Python Account
  dp_paid_py = _build_adjusted_dp(raw_paid, zeros5, zeros5, rs_paid, rp_paid, obj="SH600000")
  acc_p = Account(16000.0, dp_paid_py, fee=Fee.zero())
  _keep_and_mark(acc_p, 0, float(raw_paid[0]))
  acc_p.do_operation(1, Operation.buy(Position(1000.0, float(raw_paid[1]))))
  _keep_and_mark(acc_p, 2, float(raw_paid[2]))
  _keep_and_mark(acc_p, 3, float(raw_paid[3]))
  assert np.isclose(acc_p.position.adj_cost_price, expected_adj_paid)
  assert np.isclose(acc_p.holding_ror, expected_ror3)
  assert np.isclose(dp_paid_py.all("close")[3] / acc_p.position.adj_cost_price - 1.0, expected_ror3)
  _keep_and_mark(acc_p, 4, float(raw_paid[4]))
  assert np.isclose(acc_p.holding_ror, expected_ror4)
  assert np.isclose(dp_paid_py.all("close")[4] / acc_p.position.adj_cost_price - 1.0, expected_ror4)

  # 2. Python PortfolioAccount
  dp_paid_port = _build_adjusted_dp(raw_paid, zeros5, zeros5, rs_paid, rp_paid, obj="SH600000")
  pacc_p = PortfolioAccount(16000.0, dp_paid_port, fee=Fee.zero())
  pacc_p._buy_group(0, 1, 8000.0, float(raw_paid[1]))
  pacc_p.update_mark_to_market(2, np.array([raw_paid[2]]))
  pacc_p.apply_corporate_actions(3)
  pacc_p.update_mark_to_market(3, np.array([raw_paid[3]]))
  assert np.isclose(pacc_p.position("SH600000").adj_cost_price, expected_adj_paid)
  assert np.isclose(pacc_p.holding_ror("SH600000"), expected_ror3)
  pacc_p.update_mark_to_market(4, np.array([raw_paid[4]]))
  assert np.isclose(pacc_p.holding_ror("SH600000"), expected_ror4)

  # 3. Rust Runner (execute + signals fast-path)
  class PaidStrat(Strategy):
    def execute(self, ctx: Context):
      if ctx.bar == 1:
        ctx.buy(np.array([True]), 0.5)

  dp_paid_r = _build_adjusted_dp(raw_paid, zeros5, zeros5, rs_paid, rp_paid, obj="SH600000")
  r_p = Account(16000.0, dp_paid_r, fee=Fee.zero())
  runner_paid = Runner(r_p, exec_mode="close")
  runner_paid.run(PaidStrat())
  assert np.isclose(runner_paid.ctx.accounts[0].position.adj_cost_price, expected_adj_paid)
  assert np.isclose(runner_paid.ctx.accounts[0].holding_ror, expected_ror4)

  class PaidFastStrat(Strategy):
    def signals(self, ctx: Context):
      buy = np.array([0.0, 0.5, 0.0, 0.0, 0.0])
      sell = np.zeros(5)
      return buy, sell

  dp_paid_rf = _build_adjusted_dp(raw_paid, zeros5, zeros5, rs_paid, rp_paid, obj="SH600000")
  rf_p = Account(16000.0, dp_paid_rf, fee=Fee.zero())
  runner_paid_fast = Runner(rf_p, exec_mode="close")
  runner_paid_fast.run(PaidFastStrat())
  assert np.isclose(runner_paid_fast.ctx.accounts[0].position.adj_cost_price, expected_adj_paid)
  assert np.isclose(runner_paid_fast.ctx.accounts[0].holding_ror, expected_ror4)

  # 4. Rust PortfolioRunner
  class PaidPortStrat(PortfolioStrategy):
    def execute(self, ctx: PortfolioContext):  # type: ignore[override]
      if ctx.bar == 1:
        ctx.buy("SH600000", 0.5)

  dp_paid_rp = _build_adjusted_dp(raw_paid, zeros5, zeros5, rs_paid, rp_paid, obj="SH600000")
  rp_p = PortfolioAccount(16000.0, dp_paid_rp, fee=Fee.zero())
  PortfolioRunner(rp_p, exec_mode="close").run(PaidPortStrat())
  assert np.isclose(rp_p.position("SH600000").adj_cost_price, expected_adj_paid)
  assert np.isclose(rp_p.holding_ror("SH600000"), expected_ror4)
