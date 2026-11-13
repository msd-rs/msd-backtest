# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from types import SimpleNamespace
import numpy as np
from PIL import Image

from mbt.run.cli import build_output
from mbt.run.overview import (
  extract_holding_counts,
  extract_holding_duration_stats,
  extract_rebalance_events,
  extract_strategy_meta,
  render_strategy_overview,
  select_callout_events,
)
from mbt.run.report import Report


def _make_sample_report(
  obj: str = "PORTFOLIO",
  n_bars: int = 120,
  seed: int = 42,
) -> Report:
  rng = np.random.default_rng(seed)
  dates = np.array([20240102 + i for i in range(n_bars)], dtype=np.int64)
  # Generate realistic trading-day YYYYMMDD integers
  import datetime
  start_d = datetime.date(2024, 1, 2)
  dates = np.array(
    [int((start_d + datetime.timedelta(days=int(i * 1.45))).strftime("%Y%m%d")) for i in range(n_bars)],
    dtype=np.int64,
  )
  daily_strat = rng.normal(0.0012, 0.012, size=n_bars)
  daily_bench = rng.normal(0.0004, 0.010, size=n_bars)
  strat_ror = np.cumprod(1.0 + daily_strat) - 1.0
  bench_ror = np.cumprod(1.0 + daily_bench) - 1.0
  actions = np.zeros(n_bars, dtype=np.int32)
  for idx in (5, 25, 50, 75, 100):
    if idx < n_bars:
      actions[idx] = 1
  for idx in (35, 85):
    if idx < n_bars:
      actions[idx] = 2

  r = Report.from_dict(
    obj,
    {
      "date": dates,
      "ror": np.round(strat_ror, 4),
      "ror_hold": np.round(strat_ror, 4),
      "ror_original": np.round(bench_ror, 4),
      "benchmark_ror": np.round(bench_ror, 4),
      "benchmark_obj": "SH000300",
      "actions": actions,
      "price": 1.0 + strat_ror,
      "fw_price": 1.0 + strat_ror,
      "bw_price": 1.0 + strat_ror,
      "alpha": 0.142,
      "beta": 0.85,
      "sharpe": 1.85,
      "information_ratio": 1.25,
      "ir": 1.25,
      "sortino": 2.41,
      "treynor": 0.22,
      "max_drawdown": -0.0865,
      "calmar": 2.65,
      "win_rate": 0.625,
      "avg_win": 0.045,
      "avg_loss": -0.021,
      "profit_factor": 2.14,
      "profit": float(np.round(strat_ror[-1], 4)),
      "profit_original": float(np.round(bench_ror[-1], 4)),
      "benchmark_profit": float(np.round(bench_ror[-1], 4)),
      "fee_rate": 0.0024,
      "trade_count": 14,
      "rebalance_count": 7,
    },
  )
  r.pool_size = 50
  r.traded_objs_count = 12
  return r


class _DemoStrategy:
  """动量质量轮动策略：结合 ROE 与 20 日动量进行月度调仓，严格控制回撤。"""
  pass


class _DemoSelector:
  """基本面质量初筛器。"""
  pass


def test_extract_strategy_meta_and_overrides():
  r = _make_sample_report()
  meta = extract_strategy_meta(
    report=r,
    strategy=_DemoStrategy(),
    selector=_DemoSelector(),
    is_portfolio=True,
  )
  assert meta.title == "动量质量轮动策略"
  assert "动量质量轮动策略" in meta.brief
  assert "50" in meta.universe_headline
  assert meta.benchmark_label == "SH000300"
  assert meta.mode_label == "组合回测"
  assert meta.author == "AI Agent"

  class _AuthoredStrategy:
    M_AUTHOR = "LiJia"

  meta_authored = extract_strategy_meta(report=r, strategy=_AuthoredStrategy())
  assert meta_authored.author == "LiJia"

  meta_override = extract_strategy_meta(
    report=r,
    strategy=_DemoStrategy(),
    title="自定义标题",
    brief="自定义简介",
    universe="沪深300成分股",
  )
  assert meta_override.title == "自定义标题"
  assert meta_override.brief == "自定义简介"
  assert meta_override.universe_headline == "沪深300成分股"


def test_extract_holding_counts():
  r = _make_sample_report(n_bars=20)
  records = [
    SimpleNamespace(bar=2, obj="SH600000", action="BUY", shares=1000.0, amount=10000.0),
    SimpleNamespace(bar=5, obj="SZ000001", action="BUY", shares=500.0, amount=5000.0),
    SimpleNamespace(bar=10, obj="SH600000", action="SELL", shares=1000.0, amount=11000.0),
    SimpleNamespace(bar=15, obj="SZ000001", action="SELL", shares=500.0, amount=5200.0),
  ]
  acc = SimpleNamespace(records=records)
  counts = extract_holding_counts(r, account=acc)
  assert len(counts) == 20
  assert counts[0] == 0
  assert counts[2] == 1
  assert counts[5] == 2
  assert counts[10] == 1
  assert counts[15] == 0

  avg_d, max_d, min_d = extract_holding_duration_stats(r, account=acc)
  # SH600000: 10 - 2 = 8, SZ000001: 15 - 5 = 10 -> avg=9.0, max=10, min=8
  assert avg_d == 9.0
  assert max_d == 10
  assert min_d == 8


def test_extract_rebalance_events_and_callout_spacing():
  r = _make_sample_report(n_bars=100)
  records = [
    SimpleNamespace(bar=15, obj="SH600000", action="BUY", amount=250000.0),
    SimpleNamespace(bar=15, obj="SZ000001", action="BUY", amount=250000.0),
    SimpleNamespace(bar=17, obj="SH600519", action="BUY", amount=100000.0),
    SimpleNamespace(bar=40, obj="SH600000", action="SELL", amount=270000.0),
    SimpleNamespace(bar=40, obj="SH601318", action="BUY", amount=270000.0),
    SimpleNamespace(bar=75, obj="SZ000001", action="SELL", amount=260000.0),
    SimpleNamespace(bar=95, obj="SH601318", action="SELL", amount=290000.0),
  ]
  acc = SimpleNamespace(records=records, initial_cash=1000000.0)
  dp = SimpleNamespace(warmup_bars=10)

  events = extract_rebalance_events(r, account=acc, dp=dp)
  # bar=15 -> idx=5, bar=17 -> idx=7, bar=40 -> idx=30, bar=75 -> idx=65, bar=95 -> idx=85
  assert [e.bar_idx for e in events] == [5, 7, 30, 65, 85]
  assert "+买2" in events[0].short_label
  assert "+买1 / -卖1" == events[2].short_label

  # Dense events should be filtered to non-overlapping callouts
  callouts = select_callout_events(events, total_bars=100, max_callouts=6)
  assert 5 in [c.bar_idx for c in callouts]
  assert 7 not in [c.bar_idx for c in callouts]  # too close to 5


def test_render_strategy_overview_portfolio_and_multi(tmp_path):
  r = _make_sample_report()
  records = [
    SimpleNamespace(bar=5, obj="SH600000", action="BUY", amount=300000.0),
    SimpleNamespace(bar=5, obj="SZ000001", action="BUY", amount=300000.0),
    SimpleNamespace(bar=35, obj="SH600000", action="SELL", amount=320000.0),
    SimpleNamespace(bar=35, obj="SH600519", action="BUY", amount=320000.0),
    SimpleNamespace(bar=75, obj="SZ000001", action="SELL", amount=310000.0),
    SimpleNamespace(bar=75, obj="SH601318", action="BUY", amount=310000.0),
  ]
  acc = SimpleNamespace(records=records, initial_cash=1000000.0)
  out_png = tmp_path / "portfolio_overview.png"

  render_strategy_overview(
    [r],
    str(out_png),
    strategy=_DemoStrategy(),
    selector=_DemoSelector(),
    account=acc,
    is_portfolio=True,
  )
  assert out_png.exists()
  with Image.open(out_png) as img:
    assert img.format == "PNG"
    assert img.size == (1280, 2280)

  # Test multi-stock build_output dispatch (.png)
  r1 = _make_sample_report(obj="SH600000", seed=1)
  r2 = _make_sample_report(obj="SZ000001", seed=2)
  out_multi = tmp_path / "multi_overview.png"
  build_output([r1, r2], str(out_multi), strategy=_DemoStrategy(), is_portfolio=False)
  assert out_multi.exists()
  with Image.open(out_multi) as img:
    assert img.format == "PNG"
    assert img.size[1] > img.size[0]
    assert img.size[0] == 1280
