import os
import pytest
import numpy as np
import alpha as al
from mbt.run.msd import load_data, objs_where

MSD_HOST = os.environ.get("MSD_HOST", "http://localhost:50511")
if not os.environ.get("MSD_HOST"):
  os.environ["MSD_HOST"] = MSD_HOST


@pytest.fixture(autouse=True)
def reset_al_ctx():
  with al.Context(groups=1, flags=al.FLAG_SKIP_NAN, start=0, end=0):
    yield


def test_objs_where():
  assert objs_where(["SH600000", "SZ000001"]) == '"SH600000","SZ000001"'
  assert objs_where([]) == ""


def test_load_data_empty_objs():
  dp, objs = load_data(MSD_HOST, [], "2024-01-01", "2024-02-01")
  assert objs == []
  assert dp.bars == 0
  assert len(dp.dates) == 0


def test_load_data_real():
  try:
    dp, objs = load_data(MSD_HOST, ["SH600000"], "2025-01-01", "2025-01-15")
  except Exception as e:
    pytest.skip(f"MSD server not available: {e}")

  if len(objs) == 0:
    pytest.skip("No data returned from MSD")

  assert objs == ["SH600000"]
  assert dp.bars > 0
  assert dp.groups == 1
  assert "close" in dp.data
  assert "price" in dp.data
  assert "raw_open" in dp.data
  assert "raw_vwap" in dp.data
  assert "vwap" in dp.data
  assert "fw_price" in dp.data
  assert "bw_price" in dp.data
  assert "limited" in dp.data
  assert "limit_ratio" in dp.data
  # Verify 400-day lookback populates financial & share fields at bar 0 of Jan 2025
  assert "total_shares" in dp.data
  assert not np.isnan(dp.data["total_shares"][0])
  assert "f001" in dp.data
  assert not np.isnan(dp.data["f001"][0])
  # Default (fin_fields=None) queries * so other financial fields are also present
  assert "f002" in dp.data


def test_load_data_with_fin_fields():
  try:
    dp, objs = load_data(
      MSD_HOST, ["SH600000"], "2025-01-01", "2025-01-15", fin_fields=["f001"]
    )
  except Exception as e:
    pytest.skip(f"MSD server not available: {e}")

  if len(objs) == 0:
    pytest.skip("No data returned from MSD")

  assert objs == ["SH600000"]
  assert "f001" in dp.data
  assert not np.isnan(dp.data["f001"][0])
  # f002 should NOT be loaded when fin_fields=["f001"]
  assert "f002" not in dp.data


def test_run_request_fin_fields():
  from mbt.run.backend import RunRequest
  code = """
import mbt

class MyStrategy(mbt.Strategy):
  fin_fields = ["f001", "f007"]
  def execute(self, ctx):
    pass
"""
  req = RunRequest(code, ["SH600000"], start="2025-01-01", end="2025-01-15")
  assert getattr(req.strategy, "fin_fields", None) == ["f001", "f007"]
  assert req.fin_fields is None


def test_run_request_fin_only_year():
  from mbt.run.backend import RunRequest
  code = """
import mbt

class MyStrategy(mbt.Strategy):
  fin_only_year = True
  def execute(self, ctx):
    pass
"""
  req = RunRequest(code, ["SH600000"], start="2025-01-01", end="2025-01-15")
  assert getattr(req.strategy, "fin_only_year", False) is True

  # Also test from_json
  import json
  data = json.dumps({
    "strategy": code,
    "objs": ["SH600000"],
    "start": "2025-01-01",
    "end": "2025-01-15",
    "fin_only_year": True,
  })
  req_json = RunRequest.from_json(data)
  assert req_json.fin_only_year is True

