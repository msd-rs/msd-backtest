import pytest
import os
import polars as pl
from mbt.doctor import check_env, check_msd, run_doctor
from mbt.select.msd import MsdSelectorDataProvider


MSD_HOST = os.environ.get("MSD_HOST", "http://localhost:50511")


def test_doctor_diagnostics():
  """Verify mbt doctor runs and produces structured environment report."""
  env = check_env()
  assert "python" in env
  assert "polars" in env["packages"]

  msd_info = check_msd(MSD_HOST)
  assert msd_info["connected"] is True
  assert "tables" in msd_info

  code = run_doctor(MSD_HOST)
  assert code == 0


def test_selector_dp_load_financial_with_date():
  """Verify Polars compatibility when filtering financial records with a target date."""
  dp = MsdSelectorDataProvider(MSD_HOST, date="2024-05-10")
  # Test with date set
  df = dp.load_financial(["SH600000"], fields=["f001", "f003"], only_year=True, lastN=3)
  assert isinstance(df, pl.DataFrame)
  assert len(df) > 0
  assert "obj" in df.columns
  assert "ts" in df.columns
  assert "report_ts" in df.columns
  assert "f003" in df.columns

  # Test load_snapshot with fin_fields and date (previously crashed with InvalidOperationError)
  snap = dp.load_snapshot(["SH600000"], fin_fields=["f003"])
  assert isinstance(snap, pl.DataFrame)
  assert "f003" in snap.columns
  assert "raw_close" in snap.columns


def test_selector_dp_load_kline_with_fin_fields():
  """Verify load_kline properly aligns financial data with statutory disclosure dates."""
  dp = MsdSelectorDataProvider(MSD_HOST, date="2024-05-10")
  panel = dp.load_kline(["SH600000"], lastN=30, fin_fields=["f003"], baseline="SH000001")
  assert panel.bars == 30
  assert "f003" in panel.frame.columns
  assert "report_ts" in panel.frame.columns
  assert "raw_close" in panel.frame.columns
  # Ensure f003 is valid float
  last_f003 = panel.last("f003")[0]
  assert last_f003 > 0


def test_cli_strict_orphan_args_rejection(monkeypatch):
  """Verify that unrecognized parameters passed via -a/--args raise ValueError."""
  from mbt.run.cli import parse_args
  import sys

  test_args = [
    "mbt.run.cli",
    "strategy/double_avg.py",
    "-s",
    "SH600000",
    "-b",
    "2023-01-01",
    "-e",
    "2023-12-31",
    "-a",
    "unrecognized_parameter=999",
  ]
  monkeypatch.setattr(sys, "argv", test_args)

  with pytest.raises(ValueError, match="Unrecognized argument.*unrecognized_parameter"):
    parse_args()


def test_cli_missing_objs_descriptive_error(monkeypatch):
  """Verify that missing stocks raise actionable error instead of 'No objs provided'."""
  from mbt.run.cli import parse_args
  import sys

  test_args = [
    "mbt.run.cli",
    "strategy/double_avg.py",
    "-b",
    "2023-01-01",
    "-e",
    "2023-12-31",
  ]
  monkeypatch.setattr(sys, "argv", test_args)

  with pytest.raises(ValueError, match="No target stocks provided"):
    parse_args()


def test_portfolio_cli_rejects_implicit_first_day_pool(tmp_path, monkeypatch):
  """Verify portfolio strategy without -s raises ValueError instead of silently truncating to day-1 pool."""
  import sys
  from mbt.run.cli import parse_args

  combo_file = tmp_path / "port_combo.py"
  combo_file.write_text(
    """
import mbt

class MySelector(mbt.Selector):
  def step00(self, stocks):
    return ["SH600000", "SZ000001"]

class MyPortStrategy(mbt.PortfolioStrategy):
  def execute(self, ctx):
    pass
""",
    encoding="utf-8",
  )
  monkeypatch.setattr(
    sys,
    "argv",
    ["mbt.run.cli", str(combo_file), "-b", "2024-01-01", "-e", "2024-12-31"],
  )
  with pytest.raises(ValueError, match="requires an explicit candidate stock pool via -s/--objs"):
    parse_args()


def test_out_of_pool_and_rights_and_report_fixes():
  """Verify out-of-pool fail-fast, right_shares adj_cost_price dilution, and split-aware trade attribution."""
  import numpy as np
  from mbt import DataProvider, PortfolioAccount, PortfolioContext
  from mbt.run.backend import RunRequest
  from mbt.run.report import extract_portfolio_trades

  # 1. Cross-dimensional price anti-pattern fail-fast on load_strategy
  bad_src = """
import mbt
class BadStrat(mbt.Strategy):
  def execute(self, ctx):
    pos = ctx.position()
    ror = (ctx.close[0] - pos.price) / pos.price
"""
  with pytest.raises(ValueError, match="Cross-dimensional price arithmetic detected"):
    RunRequest.load_strategy(bad_src)

  # 2. Rights issue dilution of adj_cost_price and split-aware extract_portfolio_trades
  dp = DataProvider(
    {
      "ts": np.arange(3),
      "price": np.array([20.0, 10.0, 10.0]),
      "close": np.array([10.0, 10.0, 10.0]),
      "dividend": np.array([0.0, 0.0, 1.0]),
      "transfer_shares": np.array([0.0, 1.0, 0.0]),
      "right_shares": np.zeros(3),
      "right_price": np.zeros(3),
    },
    objs=["SZ000001"],
  )
  acc = PortfolioAccount(100000.0, dp)
  ctx = PortfolioContext(acc)

  # Out-of-pool rebalance_objs and buy must raise ValueError
  with pytest.raises(ValueError, match="not in pre-loaded pool"):
    ctx.rebalance_objs({"SH999999": 0.5})
  with pytest.raises(ValueError, match="not in the pre-loaded portfolio pool"):
    ctx.buy("SH999999", percent=0.5)

  # Buy 1000 shares @ 20.0 (cost = 20,000), then 10-for-10 split on bar 1, then 1.0 dividend on bar 2
  acc._buy_group(0, 0, 20000.0, 20.0)
  acc.apply_corporate_actions(1)
  acc.update_mark_to_market(1, np.array([10.0]))
  acc.apply_corporate_actions(2)
  acc.update_mark_to_market(2, np.array([10.0]))

  trades = extract_portfolio_trades(account=acc, dp=dp)
  assert len(trades) == 1
  # Market value (2000 * 10 = 20000) + dividends (2000 * 1 = 2000) - cost (20000) = +2000 (+10%)
  assert np.isclose(trades[0]["profit_amount"], 2000.0 - acc.deducted)
