import numpy as np
import pytest
from mbt.portfolio import PortfolioAccount
from mbt.data_provider import DataProvider
from mbt.run.report import build_portfolio_report, build_markdown_summary
from mbt.strategy import Strategy


def test_portfolio_summary_markdown_display():
  objs = ["SH000300", "SH600000", "SZ000001"]
  bars = 4
  ts_arr = np.array([
    np.datetime64("2024-01-02", "us"),
    np.datetime64("2024-01-03", "us"),
    np.datetime64("2024-01-04", "us"),
    np.datetime64("2024-01-05", "us"),
  ] * len(objs))

  price_arr = np.array([
    [100.0, 101.0, 102.0, 103.0],  # benchmark
    [10.0, 10.5, 11.0, 11.5],      # stock 1
    [20.0, 19.5, 20.0, 20.5],      # stock 2
  ]).ravel()

  data = {
    "ts": ts_arr,
    "close": price_arr,
    "price": price_arr,
    "bw_price": price_arr,
  }
  dp = DataProvider(data, objs=objs)
  account = PortfolioAccount(100000.0, dp, tradable_start=1)

  # Generate a dummy report
  reps = build_portfolio_report(dp, account)
  summary_md = build_markdown_summary(reps)

  # Check that summary has Mode: Portfolio, Pool Size, and Traded Objs
  assert "**Mode**: Portfolio (Shared Cash)" in summary_md
  assert "**Pool Size**: 2" in summary_md
  assert "**Active (Traded Objs)**: 0" in summary_md
  assert "Total Objs: 1" not in summary_md
