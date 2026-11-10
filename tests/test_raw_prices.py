import os
import pytest
import numpy as np
import polars as pl
from mbt.select.panel import Panel
from mbt.select.msd import MsdSelectorDataProvider
from mbt.run.msd import load_data

MSD_HOST = os.environ.get("MSD_HOST", "http://localhost:50511")


def test_panel_resample_with_raw_prices():
  # Create synthetic daily Panel with 4 bars for 1 stock
  raw_open = np.array([10.0, 11.0, 12.0, 13.0])
  raw_high = np.array([12.0, 13.0, 14.0, 15.0])
  raw_low = np.array([9.0, 10.0, 11.0, 12.0])
  raw_close = np.array([11.0, 12.0, 13.0, 14.0])

  # Forward-adjusted prices (e.g. after 2-for-1 split, halved)
  adj_open = raw_open * 0.5
  adj_high = raw_high * 0.5
  adj_low = raw_low * 0.5
  adj_close = raw_close * 0.5

  dates = np.array([
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"
  ], dtype="datetime64[us]")

  df = pl.DataFrame({
    "ts": dates,
    "open": adj_open,
    "high": adj_high,
    "low": adj_low,
    "close": adj_close,
    "raw_open": raw_open,
    "raw_high": raw_high,
    "raw_low": raw_low,
    "raw_close": raw_close,
    "price": raw_close,
    "volume": np.array([100.0, 200.0, 300.0, 400.0]),
    "amount": np.array([1000.0, 2200.0, 3600.0, 5200.0]),
  })

  panel = Panel(df, ["SH600000"], 4)

  # Check columns exist in panel
  assert "raw_open" in panel.frame.columns
  assert "raw_high" in panel.frame.columns
  assert "raw_low" in panel.frame.columns
  assert "raw_close" in panel.frame.columns
  assert "price" in panel.frame.columns

  # Verify last values
  assert panel.last("raw_close")[0] == 14.0
  assert panel.last("close")[0] == 7.0

  # Resample to weekly (all 4 days are in the same week: Jan 2 to Jan 5, 2024)
  w_panel = panel.resample("1W")
  assert w_panel.bars == 1
  # raw_open: first open of week = 10.0
  assert w_panel.column("raw_open")[0, 0] == 10.0
  # raw_high: max high of week = 15.0
  assert w_panel.column("raw_high")[0, 0] == 15.0
  # raw_low: min low of week = 9.0
  assert w_panel.column("raw_low")[0, 0] == 9.0
  # raw_close: last close of week = 14.0
  assert w_panel.column("raw_close")[0, 0] == 14.0
  # price alias: last of week = 14.0
  assert w_panel.column("price")[0, 0] == 14.0

  # Adjusted prices in weekly panel
  assert w_panel.column("open")[0, 0] == 5.0
  assert w_panel.column("high")[0, 0] == 7.5
  assert w_panel.column("low")[0, 0] == 4.5
  assert w_panel.column("close")[0, 0] == 7.0


def test_msd_selector_load_kline_and_snapshot_raw_prices():
  try:
    dp = MsdSelectorDataProvider(MSD_HOST)
    objs = ["SH600000"]
    snap = dp.load_snapshot(objs)
  except Exception as e:
    pytest.skip(f"MSD server not available: {e}")

  if len(snap) == 0:
    pytest.skip("No data returned from MSD")

  # Snapshot must have raw_* columns
  assert "raw_open" in snap.columns
  assert "raw_high" in snap.columns
  assert "raw_low" in snap.columns
  assert "raw_close" in snap.columns
  assert np.isclose(snap["raw_close"][0], snap["close"][0])

  # Kline must have both adjusted and raw_* columns
  panel = dp.load_kline(objs, lastN=10)
  assert "raw_open" in panel.frame.columns
  assert "raw_high" in panel.frame.columns
  assert "raw_low" in panel.frame.columns
  assert "raw_close" in panel.frame.columns
  assert "price" in panel.frame.columns
  assert "raw_vwap" in panel.frame.columns
  assert "close" in panel.frame.columns


def test_msd_backtest_load_data_raw_prices():
  try:
    dp, objs = load_data(MSD_HOST, ["SH600000"], "2025-01-01", "2025-01-15")
  except Exception as e:
    pytest.skip(f"MSD server not available: {e}")

  if len(objs) == 0:
    pytest.skip("No data returned from MSD")

  assert "raw_open" in dp.data
  assert "raw_high" in dp.data
  assert "raw_low" in dp.data
  assert "raw_close" in dp.data
  assert "price" in dp.data
  assert "raw_vwap" in dp.data
  assert "close" in dp.data
  # raw_close must equal price
  np.testing.assert_array_equal(dp.data["raw_close"], dp.data["price"])
