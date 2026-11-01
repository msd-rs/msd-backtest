from datetime import datetime
import numpy as np
import polars as pl
import pytest
from mbt.select.panel import Panel


def test_panel_resample_monthly_and_weekly():
  symbols = ["SH600000", "SZ000001"]
  # 6 trading days across 2 weeks and 2 months:
  # Week 1: 2026-09-29, 2026-09-30 (Sep)
  # Week 2: 2026-10-08, 2026-10-09 (Oct)
  # Week 3: 2026-10-12, 2026-10-13 (Oct)
  dates = [
    datetime(2026, 9, 29),
    datetime(2026, 9, 30),
    datetime(2026, 10, 8),
    datetime(2026, 10, 9),
    datetime(2026, 10, 12),
    datetime(2026, 10, 13),
  ]
  bars = len(dates)

  # Symbol 0: SH600000
  s0_open = [10.0, 10.2, 10.5, 10.8, 11.0, 11.2]
  s0_high = [10.5, 10.6, 11.0, 11.2, 11.5, 11.8]
  s0_low = [9.8, 10.0, 10.2, 10.5, 10.8, 11.0]
  s0_close = [10.2, 10.4, 10.8, 11.0, 11.3, 11.6]
  s0_vol = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0]
  s0_amt = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0]
  s0_shares = [10000.0] * bars

  # Symbol 1: SZ000001 with NaNs (e.g. suspended on day 0 and day 3)
  s1_open = [np.nan, 20.0, 20.5, np.nan, 21.0, 21.5]
  s1_high = [np.nan, 20.8, 21.0, np.nan, 21.8, 22.0]
  s1_low = [np.nan, 19.8, 20.2, np.nan, 20.8, 21.2]
  s1_close = [np.nan, 20.5, 20.8, np.nan, 21.4, 21.8]
  s1_vol = [0.0, 200.0, 300.0, 0.0, 400.0, 500.0]
  s1_amt = [0.0, 4000.0, 6000.0, 0.0, 8000.0, 10000.0]
  s1_shares = [20000.0] * bars

  df = pl.DataFrame({
    "ts": dates * 2,
    "open": s0_open + s1_open,
    "high": s0_high + s1_high,
    "low": s0_low + s1_low,
    "close": s0_close + s1_close,
    "volume": s0_vol + s1_vol,
    "amount": s0_amt + s1_amt,
    "total_shares": s0_shares + s1_shares,
  })

  panel = Panel(df, symbols, bars)

  # 1. Test Monthly Resampling ("1M")
  # Expect 2 monthly bars: 2026-09 (first 2 days) and 2026-10 (last 4 days)
  m_panel = panel.resample("1M")
  assert m_panel.bars == 2
  assert m_panel.n_symbols == 2
  assert m_panel.symbols == symbols
  assert m_panel.frame.schema == panel.frame.schema

  # Check SH600000 monthly values
  # Sep: open 10.0, close 10.4, high 10.6, low 9.8, vol 250, amt 2500
  assert m_panel.column("open")[0, 0] == 10.0
  assert m_panel.column("close")[0, 0] == 10.4
  assert m_panel.column("high")[0, 0] == 10.6
  assert m_panel.column("low")[0, 0] == 9.8
  assert m_panel.column("volume")[0, 0] == 250.0
  assert m_panel.column("amount")[0, 0] == 2500.0

  # Oct: open 10.5, close 11.6, high 11.8, low 10.2, vol 1100, amt 11000
  assert m_panel.column("open")[0, 1] == 10.5
  assert m_panel.column("close")[0, 1] == 11.6
  assert m_panel.column("high")[0, 1] == 11.8
  assert m_panel.column("low")[0, 1] == 10.2
  assert m_panel.column("volume")[0, 1] == 1100.0
  assert m_panel.column("amount")[0, 1] == 11000.0

  # Check SZ000001 (with NaNs)
  # Sep: first non-NaN open is 20.0, last non-NaN close is 20.5
  assert m_panel.column("open")[1, 0] == 20.0
  assert m_panel.column("close")[1, 0] == 20.5
  assert m_panel.column("high")[1, 0] == 20.8
  assert m_panel.column("low")[1, 0] == 19.8
  assert m_panel.column("volume")[1, 0] == 200.0

  # Oct: first non-NaN open is 20.5, last non-NaN close is 21.8
  assert m_panel.column("open")[1, 1] == 20.5
  assert m_panel.column("close")[1, 1] == 21.8
  assert m_panel.column("high")[1, 1] == 22.0
  assert m_panel.column("low")[1, 1] == 20.2
  assert m_panel.column("volume")[1, 1] == 1200.0

  # 2. Test Weekly Resampling ("1W")
  # 3 weeks: Sep 29-30, Oct 8-9, Oct 12-13
  w_panel = panel.resample("1W")
  assert w_panel.bars == 3
  assert w_panel.n_symbols == 2

  # Week 1 (Sep 29-30)
  assert w_panel.column("open")[0, 0] == 10.0
  assert w_panel.column("close")[0, 0] == 10.4
  # Week 2 (Oct 8-9)
  assert w_panel.column("open")[0, 1] == 10.5
  assert w_panel.column("close")[0, 1] == 11.0
  # Week 3 (Oct 12-13)
  assert w_panel.column("open")[0, 2] == 11.0
  assert w_panel.column("close")[0, 2] == 11.6

  # 3. Test Invalid Frequency Error
  with pytest.raises(ValueError):
    panel.resample("invalid_freq_xyz")

