import os
import pytest
import numpy as np
import alpha as al
from mbt.run.msd import load_data, objs_where

MSD_HOST = os.environ.get("MSD_HOST", "http://localhost:50511")


@pytest.fixture(autouse=True)
def reset_al_ctx():
  yield
  al.set_ctx(groups=1, flags=al.FLAG_SKIP_NAN, start=0, end=0)


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
    dp, objs = load_data(MSD_HOST, ["SH600000"], "2024-01-01", "2024-01-10")
  except Exception as e:
    pytest.skip(f"MSD server not available: {e}")

  if len(objs) == 0:
    pytest.skip("No data returned from MSD")

  assert objs == ["SH600000"]
  assert dp.bars > 0
  assert dp.groups == 1
  assert "close" in dp.data
  assert "price" in dp.data
  assert "fw_price" in dp.data
  assert "bw_price" in dp.data
  assert "limited" in dp.data
  assert "limit_ratio" in dp.data
