from pathlib import Path
import numpy as np
from mbt import Account, DataProvider, Runner
from mbt.account import ACTION_BUY, ACTION_BUY_NO_CASH, ACTION_SELL
from mbt.run.backend import RunRequest


def test_holding_ror_grid_entry_addon_takeprofit_and_reentry():
  root = Path(__file__).resolve().parent.parent
  strat = RunRequest.load_strategy(str(root / "strategy" / "holding_ror_grid.py"))

  # Build synthetic series (using ma_period=3 for compact deterministic test)
  strat.ma_period = 3
  strat.warmup_bars = 0

  # Bars:
  # 0-2: 10.0, 10.0, 10.0 -> MA3=10.0 (not below MA)
  # 3: 9.0 (< MA3=9.67) -> initial buy 5% of 100,000 = 5,000 -> 500 shares @ 9.0 = 4,500, cash=95,500
  # 4: 8.1 -> holding_ror = (8.1 - 9.0)/9.0 = -10% -> add-on #1: 5% of 95,500 = 4,775 -> 500 shares @ 8.1 = 4,050
  #       total 1,000 shares, avg cost = 8.55, cash = 91,450
  # 5: 6.84 -> holding_ror = (6.84 - 8.55)/8.55 = -20% -> add-on #2: 5% of 91,450 = 4,572.5 -> 600 shares @ 6.84 = 4,104
  #       total 1,600 shares, total cost = 12,654, avg cost = 7.90875
  # 6: 9.50 -> holding_ror = (9.50 - 7.90875)/7.90875 = +20.12% >= +20% -> sell 100%!
  # 7: 11.0 (> MA3=9.11) -> stays flat (does not re-enter above MA)
  # 8: 8.0 (< MA3=9.50) -> re-enters!
  prices = np.array([10.0, 10.0, 10.0, 9.0, 8.1, 6.84, 9.50, 11.0, 8.0], dtype=np.float64)
  n = len(prices)
  zeros = np.zeros(n, dtype=np.float64)

  dp = DataProvider(
    {
      "ts": np.arange(n, dtype=np.int64) * 86400,
      "open": prices,
      "high": prices,
      "low": prices,
      "close": prices,
      "price": prices,
      "dividend": zeros,
      "transfer_shares": zeros,
      "right_shares": zeros,
      "right_price": zeros,
    },
    objs=["SH600000"],
  )

  acc = Account(100_000.0, dp)
  runner = Runner(acc)
  runner.run(strat)

  actions = dp.all("actions")
  assert actions[3] == ACTION_BUY
  assert actions[4] == ACTION_BUY
  assert actions[5] == ACTION_BUY
  assert actions[6] == ACTION_SELL
  assert actions[7] == 0  # stays out above MA
  assert actions[8] == ACTION_BUY  # re-enters below MA

  diag = runner.diagnostics
  assert diag == {"initial_entries": 2, "add_on_buys": 2, "take_profit_exits": 1}


def test_holding_ror_grid_split_safety_and_insufficient_cash():
  root = Path(__file__).resolve().parent.parent
  strat = RunRequest.load_strategy(str(root / "strategy" / "holding_ror_grid.py"))
  strat.ma_period = 2
  strat.warmup_bars = 0

  # Initial cash 25,000, buy_percent=5% -> 1,250 budget.
  # Bar 1: close=10.0 (< MA2=11.0) -> buys 100 shares @ 10.0 (1,000 spent, cash=24,000)
  # Bar 2: 10-for-10 split (transfer_shares=1.0), raw price=5.0, fw_close=10.0
  #        True holding_ror is 0% (NOT -50%!), so MUST NOT trigger add-on buy!
  # Bar 3: raw price=4.5, fw_close=9.0 -> holding_ror = -10% -> triggers add-on buy!
  raw_price = np.array([12.0, 10.0, 5.0, 4.5], dtype=np.float64)
  fw_close = np.array([12.0, 10.0, 10.0, 9.0], dtype=np.float64)
  trans = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
  zeros = np.zeros(4, dtype=np.float64)

  dp = DataProvider(
    {
      "ts": np.arange(4, dtype=np.int64) * 86400,
      "open": fw_close,
      "high": fw_close,
      "low": fw_close,
      "close": fw_close,
      "price": raw_price,
      "dividend": zeros,
      "transfer_shares": trans,
      "right_shares": zeros,
      "right_price": zeros,
    },
    objs=["SH600000"],
  )

  runner = Runner(Account(25_000.0, dp))
  runner.run(strat)
  actions = dp.all("actions")
  assert actions[1] == ACTION_BUY
  assert actions[2] == 0  # split does NOT trigger false add-on
  assert actions[3] == ACTION_BUY  # true -10% triggers add-on

  # Verify insufficient cash case: initial cash 1,500 -> 5% = 75 < 100*10=1000 -> ACTION_BUY_NO_CASH
  dp_low = DataProvider(
    {
      "ts": np.arange(3, dtype=np.int64) * 86400,
      "open": np.array([12.0, 10.0, 10.0]),
      "close": np.array([12.0, 10.0, 10.0]),
      "price": np.array([12.0, 10.0, 10.0]),
      "dividend": np.zeros(3),
      "transfer_shares": np.zeros(3),
      "right_shares": np.zeros(3),
      "right_price": np.zeros(3),
    },
    objs=["SH600000"],
  )
  strat_low = RunRequest.load_strategy(str(root / "strategy" / "holding_ror_grid.py"))
  strat_low.ma_period = 2
  strat_low.warmup_bars = 0
  runner_low = Runner(Account(1_500.0, dp_low))
  runner_low.run(strat_low)
  assert dp_low.all("actions")[1] == ACTION_BUY_NO_CASH


def test_context_vectorized_holding_apis():
  from mbt import PortfolioAccount, PortfolioRunner, Strategy

  ts = np.tile(np.arange(4, dtype=np.int64) * 86400, 2)
  prices = np.array([10.0, 10.0, 12.0, 12.0, 20.0, 20.0, 18.0, 18.0])
  zeros = np.zeros(8, dtype=np.float64)

  dp = DataProvider(
    {
      "ts": ts,
      "open": prices,
      "close": prices,
      "price": prices,
      "dividend": zeros,
      "transfer_shares": zeros,
      "right_shares": zeros,
      "right_price": zeros,
    },
    objs=["A", "B"],
  )

  seen = []

  class ProbeStrat(Strategy):
    def execute(self, ctx):
      if ctx.bar == 1:
        ctx.buy(np.array([True, False]), percent=1.0)
      elif ctx.bar == 2:
        seen.append((ctx.holding_shares().copy(), ctx.holding_rors().copy()))

  Runner(Account(10_000.0, dp)).run(ProbeStrat())
  assert len(seen) == 1
  assert seen[0][0].tolist() == [1000.0, 0.0]
  assert np.allclose(seen[0][1], [0.2, 0.0])

  seen.clear()
  p_acc = PortfolioAccount(10_000.0, dp)
  PortfolioRunner(p_acc).run(ProbeStrat())
  assert len(seen) == 1
  assert seen[0][0].tolist() == [1000.0, 0.0]
  assert np.allclose(seen[0][1], [0.2, 0.0])


def test_buy_no_cash_guard_and_no_rejected_records(caplog):
  import logging
  from mbt.run.report import build_markdown_summary, build_report

  dp = DataProvider(
    {
      "ts": np.arange(4, dtype=np.int64) * 86400,
      "open": np.array([20.0, 18.0, 17.0, 16.0]),
      "close": np.array([20.0, 18.0, 17.0, 16.0]),
      "price": np.array([20.0, 18.0, 17.0, 16.0]),
      "bw_price": np.array([20.0, 18.0, 17.0, 16.0]),
      "dividend": np.zeros(4),
      "transfer_shares": np.zeros(4),
      "right_shares": np.zeros(4),
      "right_price": np.zeros(4),
    },
    objs=["SH600000"],
  )
  root = Path(__file__).resolve().parent.parent
  strat = RunRequest.load_strategy(str(root / "strategy" / "holding_ror_grid.py"))
  strat.ma_period = 2
  strat.warmup_bars = 0

  acc = Account(10_000.0, dp)  # 5% = 500 < 100 * 16 = 1600 -> rejected every bar
  runner = Runner(acc)
  with caplog.at_level(logging.WARNING, logger="mbt.run.runner"):
    runner.run(strat)

  assert dp.all("actions")[1] == ACTION_BUY_NO_CASH
  # Rejected orders must NOT bloat Account.records
  assert len(runner.ctx.accounts[0].records) == 0
  assert any("BUY_NO_CASH GUARD" in rec.message for rec in caplog.records)

  reports = build_report(dp, runner.ctx.accounts)
  md = build_markdown_summary(reports, account=acc)
  assert "BUY_NO_CASH" in md
  assert "> [!WARNING]" in md
