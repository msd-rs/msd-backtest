import numpy as np
import polars as pl
import pytest
from mbt.select.panel import Panel


def test_panel_zero_copy_slicing():
  """Verify that Panel.slice shares memory with original 2D view and doesn't eagerly create DataFrame."""
  n_objs = 3
  bars = 10
  objs = ["SH600000", "SZ000001", "SZ000002"]

  # Create views directly with numpy
  raw_close = np.arange(n_objs * bars, dtype=np.float64).reshape(n_objs, bars)
  raw_open = raw_close + 0.5

  p = Panel({"raw_close": raw_close, "raw_open": raw_open}, objs, bars)

  # Check that _frame is initially None (Lazy Materialization)
  assert p._frame is None
  assert p.columns == ["raw_close", "raw_open"]
  assert p.bars == bars
  assert p.n_objs == n_objs

  # Slice from bar 3 to bar 8 (5 bars)
  sub = p.slice(3, 8)
  assert sub.bars == 5
  assert sub.n_objs == 3
  assert sub._frame is None

  # Verify memory is shared (zero copy!)
  sub_close = sub.column("raw_close")
  assert np.shares_memory(sub_close, raw_close)
  assert sub_close.shape == (3, 5)
  assert np.array_equal(sub_close, raw_close[:, 3:8])

  # Slicing with specific objs
  sub_obj = p.slice(3, 8, objs=["SZ000001"])
  assert sub_obj.n_objs == 1
  assert sub_obj.objs == ["SZ000001"]
  assert np.array_equal(sub_obj.column("raw_close")[0], raw_close[1, 3:8])

  # Calling .frame triggers materialization on demand
  df = sub.frame
  assert isinstance(df, pl.DataFrame)
  assert len(df) == 3 * 5
  assert "raw_close" in df.columns
  assert "raw_open" in df.columns


def test_panel_buffer_on_strided_view():
  """Verify buffer() returns 1D contiguous array even when view is strided."""
  n_objs = 2
  bars = 6
  objs = ["A", "B"]
  data = np.arange(12, dtype=np.float64).reshape(2, 6)
  p = Panel({"val": data}, objs, 6)

  sub = p.slice(2, 5)  # 3 bars per obj, strided!
  assert not sub.column("val").flags.c_contiguous

  buf = sub.buffer("val")
  assert buf.flags.c_contiguous
  assert len(buf) == 2 * 3
  expected = np.concatenate([data[0, 2:5], data[1, 2:5]])
  assert np.array_equal(buf, expected)
