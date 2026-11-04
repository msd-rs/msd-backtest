from mbt.select.msd import MsdSelectorDataProvider
import os
import time

MSD_HOST = os.environ.get("MSD_HOST", "http://localhost:50511")


def bench_load_financial():
  dp = MsdSelectorDataProvider(MSD_HOST)
  fields: list[str] = "f001,f007,f095,f130,f203".split(',')
  t1 = time.time()
  dp.load_financial(["*"], fields, False, 8)
  print(f"load used {time.time() - t1:.3f}s")


def test_load_financial_long_dataframe():
  import polars as pl
  dp = MsdSelectorDataProvider(MSD_HOST)
  fields = ["f001", "f007"]
  symbols = ["SH600000", "SZ000001"]

  df = dp.load_financial(symbols, fields=fields, only_year=True, lastN=3)
  assert isinstance(df, pl.DataFrame)
  assert "obj" in df.columns
  assert "ts" in df.columns
  for f in fields:
    assert f in df.columns
  assert df.columns[0] == "obj"
  assert len(df) > 0
  assert set(df["obj"].unique().to_list()).issubset(set(symbols))


def test_load_financial_empty():
  import polars as pl
  dp = MsdSelectorDataProvider(MSD_HOST)
  df = dp.load_financial([], fields=["f001"])
  assert isinstance(df, pl.DataFrame)
  assert len(df) == 0


def test_load_snapshot_with_fin():
  dp = MsdSelectorDataProvider(MSD_HOST)
  symbols = ["SH600000", "SZ000001"]
  snap = dp.load_snapshot(symbols, fin_fields=["f001"])
  assert "obj" in snap.columns
  assert "f_ts" in snap.columns
  assert "f001" in snap.columns
  assert len(snap) == len(symbols)


if __name__ == "__main__":
  bench_load_financial()