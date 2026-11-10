import datetime
import numpy as np
import polars as pl
import pytest
from mbt.shared import to_statutory_disclosure_date


def test_statutory_disclosure_date():
  dates = pl.Series(
    "report_ts",
    [
      datetime.datetime(2023, 3, 31),
      datetime.datetime(2023, 6, 30),
      datetime.datetime(2023, 9, 30),
      datetime.datetime(2023, 12, 31),
    ],
  )
  ann = to_statutory_disclosure_date(dates)
  ann_str = [str(d)[:10] for d in ann.to_list()]
  assert ann_str == ["2023-04-30", "2023-08-31", "2023-10-31", "2024-04-30"]


def test_financial_empty_range_safe_alignment():
  """Test that objects with empty financial ranges do not raise IndexError."""
  objs = ["SH600000", "EMPTY_OBJ"]
  bars = 5
  kline_ts = np.array([
    np.datetime64("2024-05-01", "us") + np.timedelta64(i, "D")
    for _ in range(len(objs))
    for i in range(bars)
  ])

  # SH600000 has 1 report, EMPTY_OBJ has 0 reports (st == end_idx)
  rep_ts_raw = np.array([np.datetime64("2023-12-31", "us")])
  ann_ts_raw = np.array([np.datetime64("2024-04-30", "us")])
  f001_raw = np.array([1.5])
  ranges = [("SH600000", 0, 1), ("EMPTY_OBJ", 1, 1)]

  data = {
    "ts": kline_ts,
    "f001": np.full(len(objs) * bars, np.nan),
    "report_ts": np.full(len(objs) * bars, np.datetime64("NaT", "us"), dtype="datetime64[us]"),
  }

  range_map = {r[0]: (r[1], r[2]) for r in ranges}
  for g, obj in enumerate(objs):
    kline_slice = slice(g * bars, (g + 1) * bars)
    if obj not in range_map:
      continue
    st, end_idx = range_map[obj]
    if st >= end_idx:
      continue
    obj_ann_ts = ann_ts_raw[st:end_idx]
    if len(obj_ann_ts) == 0:
      continue
    obj_rep_ts = rep_ts_raw[st:end_idx]
    kline_ts_slice = data["ts"][kline_slice]

    idx = np.searchsorted(obj_ann_ts, kline_ts_slice, side="right") - 1
    valid = idx >= 0
    safe_idx = np.maximum(idx, 0)

    val = f001_raw[st:end_idx]
    if len(val) == 0:
      continue
    data["f001"][kline_slice] = np.where(valid, val[safe_idx], np.nan)
    data["report_ts"][kline_slice] = np.where(valid, obj_rep_ts[safe_idx], np.datetime64("NaT", "us"))

  # SH600000 should be aligned to 1.5
  assert np.all(data["f001"][:bars] == 1.5)
  # EMPTY_OBJ should remain all NaN
  assert np.all(np.isnan(data["f001"][bars:]))
