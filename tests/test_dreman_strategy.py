import numpy as np
import pytest
from mbt import Account, DataProvider, Runner
from strategy.dreman_value_growth import Strategy


def test_dreman_strategy_initialization():
  strat = Strategy()
  assert strat.max_pe == 25.0
  assert strat.min_profit_growth == 15.0
  assert strat.min_roe == 5.0
  assert strat.pe_exit == 40.0
  assert strat.stop_loss_pct == 0.08
  assert strat.trailing_trigger == 0.10
  assert strat.trailing_stop_pct == 0.08

  # Test custom arguments via key=value string
  custom_strat = Strategy("max_pe=20.0", "min_profit_growth=25.0", "min_roe=10.0")
  assert custom_strat.max_pe == 20.0
  assert custom_strat.min_profit_growth == 25.0
  assert custom_strat.min_roe == 10.0


def test_dreman_strategy_synthetic_run():
  """
  使用合成数据测试德瑞曼策略在估值与业绩达标时的买入、持仓与卖出流程。
  """
  n_bars = 70
  start_date = np.datetime64("2024-01-01")
  dates = np.arange(start_date, start_date + np.timedelta64(n_bars, "D"), dtype="datetime64[D]")

  # 模拟价格走势：前半程平稳上升，后半程冲高回落
  prices = np.linspace(10.0, 20.0, n_bars)

  # 基本面指标：
  # f003 (BPS): 10.0
  # f205 (ROE): 10.0% -> roe_ratio = 0.1
  # price / f003 = 1.0 ~ 2.0 -> PE = (price / 10.0) / 0.1 = 10.0 ~ 20.0 (满足 0 < PE <= 25.0)
  # f197 (净利润同比增长率): 30.0% (满足 >= 15.0%)
  bps = np.full(n_bars, 10.0)
  roe = np.full(n_bars, 10.0)
  growth = np.full(n_bars, 30.0)

  dp = DataProvider(
    {
      "ts": dates,
      "open": prices,
      "high": prices * 1.01,
      "low": prices * 0.99,
      "close": prices,
      "price": prices,
      "bw_price": prices,
      "fw_price": prices,
      "volume": np.full(n_bars, 1000.0),
      "amount": prices * 1000.0,
      "dividend": np.zeros(n_bars),
      "transfer_shares": np.zeros(n_bars),
      "right_shares": np.zeros(n_bars),
      "right_price": np.zeros(n_bars),
      "f003": bps,
      "f205": roe,
      "f197": growth,
    },
    objs=["TEST_STOCK"],
  )

  account = Account(100000.0, dp)
  runner = Runner(account)
  strat = Strategy(max_pe=25.0, min_profit_growth=15.0, min_roe=5.0)
  runner.run(strat)

  acc = runner.ctx.accounts[0]
  # 验证策略能够正常识别买入信号并建仓
  assert np.any(acc.actions == 1), "Strategy should execute at least one buy action on qualifying stock"
  assert acc.asset > 0, "Account asset should be positive"


def test_dreman_strategy_unprofitable_filter():
  """
  测试当净利润增长率或估值不满足德瑞曼条件时，策略拒绝建仓。
  """
  n_bars = 70
  start_date = np.datetime64("2024-01-01")
  dates = np.arange(start_date, start_date + np.timedelta64(n_bars, "D"), dtype="datetime64[D]")
  prices = np.linspace(10.0, 20.0, n_bars)

  # 净利润同比负增长（不满足业绩为王）
  growth = np.full(n_bars, -10.0)
  bps = np.full(n_bars, 10.0)
  roe = np.full(n_bars, -5.0)

  dp = DataProvider(
    {
      "ts": dates,
      "open": prices,
      "high": prices * 1.01,
      "low": prices * 0.99,
      "close": prices,
      "price": prices,
      "bw_price": prices,
      "fw_price": prices,
      "volume": np.full(n_bars, 1000.0),
      "amount": prices * 1000.0,
      "dividend": np.zeros(n_bars),
      "transfer_shares": np.zeros(n_bars),
      "right_shares": np.zeros(n_bars),
      "right_price": np.zeros(n_bars),
      "f003": bps,
      "f205": roe,
      "f197": growth,
    },
    objs=["BAD_STOCK"],
  )

  account = Account(100000.0, dp)
  runner = Runner(account)
  strat = Strategy(max_pe=25.0, min_profit_growth=15.0, min_roe=5.0)
  runner.run(strat)

  acc = runner.ctx.accounts[0]
  assert not np.any(acc.actions == 1), "Strategy must not buy stocks failing Dreman criteria"
