from datetime import datetime, timedelta
import numpy as np
import polars as pl
import pytest
from mbt.select import SelectorDataProvider, Panel
from strategy.volume_breakout_monthly_yang_selector import VolumeBreakoutMonthlyYangSelector


class MockSelectorDataProvider(SelectorDataProvider):
  def __init__(self, panel: Panel, snap: pl.DataFrame):
    super().__init__()
    self._panel = panel
    self._snap = snap

  def load_snapshot(self, symbols: list[str], fin_fields: list[str] = [], fin_only_year: bool = True) -> pl.DataFrame:
    return self._snap.filter(pl.col("obj").is_in(symbols))

  def load_kline(
    self,
    symbols: list[str],
    lastN: int | list[int] = 100,
    fin_fields: list[str] = [],
    fin_only_year: bool = True,
    baseline: str | None = None,
  ) -> Panel:
    # Filter panel by symbols
    sub_symbols = [s for s in symbols if s in self._panel.symbols]
    bars = self._panel.bars
    rows = []
    for s in sub_symbols:
      idx = self._panel.pick(s)
      rows.append(self._panel.frame.slice(idx * bars, bars))
    df = pl.concat(rows)
    return Panel(df, sub_symbols, bars)


def test_volume_breakout_monthly_yang_selector():
  symbols = ["SZ000001", "SZ000002", "SZ000003", "SZ000004"]
  bars = 80

  # Construct dates covering 4 months: 2026-07 to 2026-10
  # 20 bars per month
  dates = []
  for month in [7, 8, 9, 10]:
    for day in range(1, 21):
      dates.append(datetime(2026, month, day))

  dfs = []
  for sym in symbols:
    # Default base prices and volume
    open_p = np.full(bars, 10.0)
    high_p = np.full(bars, 10.5)
    low_p = np.full(bars, 9.5)
    close_p = np.full(bars, 10.2)
    vol = np.full(bars, 1000.0)
    amt = vol * close_p

    # Set 3 target months (Aug, Sep, Oct)
    # Aug: bars 20..39 -> Month 8
    # Sep: bars 40..59 -> Month 9
    # Oct: bars 60..79 -> Month 10

    if sym == "SZ000001":
      # Passes all:
      # Month 8: open 10.0, close 11.0 (Yang)
      open_p[20] = 10.0
      close_p[39] = 11.0
      # Month 9: open 11.0, close 12.0 (Yang)
      open_p[40] = 11.0
      close_p[59] = 12.0
      # Month 10: open 12.0, close 13.0 (Yang)
      open_p[60] = 12.0
      close_p[78] = 12.5
      close_p[79] = 14.0  # today
      open_p[79] = 13.0
      high_p[79] = 14.2
      # Preceding 20-day high is 12.8
      high_p[60:79] = 12.8
      # Volume surge today: 2500 vs 5-day MA 1000
      vol[79] = 2500.0

    elif sym == "SZ000002":
      # Fails Month 9 Yang (Month 9 is Yin: open 12.0, close 11.0)
      open_p[20] = 10.0
      close_p[39] = 11.0
      open_p[40] = 12.0
      close_p[59] = 11.0  # Yin!
      open_p[60] = 11.0
      close_p[79] = 13.0
      high_p[60:79] = 12.0
      vol[79] = 2500.0

    elif sym == "SZ000003":
      # Passes monthly Yang, but fails breakout (close < prev_high)
      open_p[20] = 10.0
      close_p[39] = 11.0
      open_p[40] = 11.0
      close_p[59] = 12.0
      open_p[60] = 12.0
      close_p[79] = 12.5  # close is 12.5
      high_p[60:79] = 15.0  # prev high is 15.0 -> Not a breakout!
      vol[79] = 2500.0

    elif sym == "SZ000004":
      # Passes monthly Yang & breakout, but fails volume surge (vol_ratio < 1.5)
      open_p[20] = 10.0
      close_p[39] = 11.0
      open_p[40] = 11.0
      close_p[59] = 12.0
      open_p[60] = 12.0
      close_p[79] = 14.0
      high_p[60:79] = 12.8
      vol[79] = 1100.0  # 1100 vs MA 1000 -> ratio 1.1 < 1.5!

    df_sym = pl.DataFrame({
      "obj": [sym] * bars,
      "ts": dates,
      "open": open_p,
      "high": high_p,
      "low": low_p,
      "close": close_p,
      "volume": vol,
      "amount": amt,
    })
    dfs.append(df_sym)

  full_frame = pl.concat(dfs)
  panel = Panel(full_frame, symbols, bars)

  snap = pl.DataFrame({
    "obj": symbols,
    "open": [13.0, 12.0, 12.0, 13.0],
    "close": [14.0, 13.0, 12.5, 14.0],
    "volume": [2500.0, 2500.0, 2500.0, 1100.0],
    "amount": [35000.0, 32500.0, 31250.0, 15400.0],
  })

  dp = MockSelectorDataProvider(panel, snap)
  selector = VolumeBreakoutMonthlyYangSelector(
    breakout_days=20,
    volume_ratio=1.5,
    volume_ma_days=5,
    monthly_yang_count=3,
    today_yang=True,
  )

  selected = selector.execute(dp, symbols)

  assert selected == ["SZ000001"]
  assert selector.factors is not None
  assert len(selector.factors) == 1
  assert selector.factors[0, "obj"] == "SZ000001"
  assert selector.factors[0, "vol_ratio"] >= 1.5
  assert selector.factors[0, "breakout_pct"] > 0

