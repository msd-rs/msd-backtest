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
