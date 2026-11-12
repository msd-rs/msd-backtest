import os
import numpy as np
import pytest
import polars as pl
import mbt
from mbt import strategy_templates, DataProvider, PortfolioAccount, PortfolioRunner, Runner, Account
from mbt.run.backend import RunRequest
from mbt.run.validator import check_strategy_source
from mbt.strategy_templates.fundamental_timing import (
  FundamentalTimingSelector,
  FundamentalTimingStrategy,
)
from mbt.strategy_templates.multifactor_rotation import (
  MultiFactorRotationStrategy,
)
from mbt.strategy_templates.cta_momentum import (
  CtaMomentumStrategy,
)


def _make_dummy_stock_data(n_bars: int, prices: np.ndarray, dates: np.ndarray) -> dict:
  """创建符合 DataProvider 标准字段的合成行情数据."""
  return {
    "ts": dates,
    "open": prices,
    "high": prices * 1.02,
    "low": prices * 0.98,
    "close": prices,
    "price": prices,
    "raw_close": prices,
    "bw_price": prices,
    "fw_price": prices,
    "volume": np.full(n_bars, 1000.0),
    "amount": prices * 1000.0,
    "dividend": np.zeros(n_bars),
    "transfer_shares": np.zeros(n_bars),
    "right_shares": np.zeros(n_bars),
    "right_price": np.zeros(n_bars),
  }


def test_template_registry():
  templates = strategy_templates.list_templates()
  assert "fundamental_timing" in templates
  assert "multifactor_rotation" in templates
  assert "cta_momentum" in templates

  for t_name in templates:
    path = strategy_templates.get_template_path(t_name)
    assert os.path.isfile(path)
    with open(path, "r", encoding="utf-8") as f:
      content = f.read()
      assert len(content) > 0


def test_template_anti_pattern_linting():
  """验证所有官方模版源码均无跨维度混合计算等任何反模式警告."""
  for t_name in strategy_templates.list_templates():
    path = strategy_templates.get_template_path(t_name)
    with open(path, "r", encoding="utf-8") as f:
      code = f.read()
    warnings = check_strategy_source(code, filename=path)
    assert len(warnings) == 0, f"Template {t_name} generated AST linting warnings: {warnings}"


def test_fundamental_timing_template_synthetic():
  """验证模版 ① 基本面筛选 + 趋势择时在合成数据上的执行流."""
  n_bars = 80
  start_date = np.datetime64("2024-01-01")
  dates = np.arange(start_date, start_date + np.timedelta64(n_bars, "D"), dtype="datetime64[D]")

  prices = np.linspace(10.0, 20.0, n_bars)
  data = _make_dummy_stock_data(n_bars, prices, dates)
  data["f003"] = np.full(n_bars, 8.0)
  data["f205"] = np.full(n_bars, 12.0)
  data["f197"] = np.full(n_bars, 20.0)

  dp = DataProvider(data, objs=["SH600000"])

  account = PortfolioAccount(1000000.0, dp)
  strat = FundamentalTimingStrategy(
    fast_period=5,
    slow_period=20,
    trend_period=30,
    stop_loss_pct=0.08,
    trailing_trigger=0.10,
    max_positions=2,
  )

  runner = PortfolioRunner(account)
  runner.run(strat)

  diag = strat.finalize(runner.ctx)
  assert "exit_stop_loss_count" in diag
  assert "exit_trailing_stop_count" in diag
  assert "exit_trend_break_count" in diag


def test_multifactor_rotation_template_synthetic():
  """验证模版 ② 多因子全市场轮动在多标的合成数据上的执行流."""
  n_bars = 90
  start_date = np.datetime64("2024-01-01")
  dates = np.arange(start_date, start_date + np.timedelta64(n_bars, "D"), dtype="datetime64[D]")

  objs = ["SH600000", "SZ000001", "SH600519"]
  p1 = np.linspace(10.0, 15.0, n_bars)
  p2 = np.linspace(20.0, 15.0, n_bars)
  p3 = np.linspace(50.0, 80.0, n_bars)

  all_dates = np.concatenate([dates, dates, dates])
  all_opens = np.concatenate([p1, p2, p3])
  all_highs = np.concatenate([p1 * 1.01, p2 * 1.01, p3 * 1.01])
  all_lows = np.concatenate([p1 * 0.99, p2 * 0.99, p3 * 0.99])
  all_closes = np.concatenate([p1, p2, p3])
  all_f003 = np.concatenate([np.full(n_bars, 8.0), np.full(n_bars, 10.0), np.full(n_bars, 25.0)])
  all_f205 = np.concatenate([np.full(n_bars, 10.0), np.full(n_bars, 5.0), np.full(n_bars, 20.0)])

  data = {
    "ts": all_dates,
    "open": all_opens,
    "high": all_highs,
    "low": all_lows,
    "close": all_closes,
    "price": all_closes,
    "raw_close": all_closes,
    "bw_price": all_closes,
    "fw_price": all_closes,
    "volume": np.full(n_bars * 3, 1000.0),
    "amount": all_closes * 1000.0,
    "dividend": np.zeros(n_bars * 3),
    "transfer_shares": np.zeros(n_bars * 3),
    "right_shares": np.zeros(n_bars * 3),
    "right_price": np.zeros(n_bars * 3),
    "f003": all_f003,
    "f205": all_f205,
  }

  dp = DataProvider(data, objs=objs)

  account = PortfolioAccount(1000000.0, dp)
  strat = MultiFactorRotationStrategy(
    top_k=2,
    rebalance_interval=10,
    weight_momentum=0.5,
    weight_value=0.25,
    weight_quality=0.25,
  )

  runner = PortfolioRunner(account)
  runner.run(strat)

  diag = strat.finalize(runner.ctx)
  assert diag["rebalance_count"] > 0
  assert diag["top_k"] == 2


def test_cta_momentum_template_synthetic():
  """验证模版 ③ CTA 动量突破与 ATR 跟踪止损在合成数据上的执行流."""
  n_bars = 70
  start_date = np.datetime64("2024-01-01")
  dates = np.arange(start_date, start_date + np.timedelta64(n_bars, "D"), dtype="datetime64[D]")

  p = np.full(n_bars, 10.0)
  p[25:50] = np.linspace(10.0, 18.0, 25)
  p[50:] = np.linspace(18.0, 12.0, 20)

  data = _make_dummy_stock_data(n_bars, p, dates)
  dp = DataProvider(data, objs=["SH600000"])

  account = Account(100000.0, dp)
  strat = CtaMomentumStrategy(
    channel_period=15,
    exit_channel_period=8,
    atr_period=10,
    atr_stop_multiplier=1.5,
    fixed_stop_loss=0.05,
  )

  runner = Runner(account)
  runner.run(strat)

  diag = strat.finalize(runner.ctx)
  assert "exit_fixed_stop_count" in diag
  assert "exit_atr_trailing_count" in diag
  assert "exit_channel_exit_count" in diag


def test_load_strategy_from_template_path():
  """验证 RunRequest.load_strategy 能从模版文件路径直接加载并正确绑定参数."""
  cta_path = strategy_templates.get_template_path("cta_momentum")
  strat = RunRequest.load_strategy(cta_path, args={"channel_period": 25, "fixed_stop_loss": 0.08})
  assert isinstance(strat, mbt.Strategy)
  assert strat.__class__.__name__ == "CtaMomentumStrategy"
  assert strat.channel_period == 25
  assert strat.fixed_stop_loss == 0.08
