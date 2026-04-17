from mbt import DataProvider
import pandas as pd
import numpy as np


def uniform_ror(a: np.ndarray) -> np.ndarray:
  base = a[0]
  if base == 0:
    return a
  return ((a - base) / base * 1000).astype(np.int32)


def build_report(dp: DataProvider) -> pd.DataFrame:
  symbols = np.repeat(dp.symbols, dp.bars)
  dates = dp.dates
  df = pd.DataFrame(
    {
      "symbol": symbols,
      "date": dates,
      "close": (dp.all("close") * 1000).astype(np.int64) / 1000.0,
      "close_ror": uniform_ror(dp.all("close")),
      "ror": (dp.all("ror") * 1000).astype(np.int32),
      "ror_hold": (dp.all("ror_hold") * 1000).astype(np.int32),
      "actions": dp.all("actions"),
    }
  )
  return df


def build_json_report(dp: DataProvider, df: pd.DataFrame | None = None) -> dict:
  """Helper to build a JSON report from DataProvider"""

  if df is None:
    df = build_report(dp)

  data = {}
  for i, symbol in enumerate(dp.symbols):
    d = df.iloc[i * dp.bars : (i + 1) * dp.bars].to_dict(orient="list")  # type: ignore
    del d["symbol"]
    # Format dates as strings
    d["date"] = [t.strftime("%Y-%m-%d %X").rstrip(" 00:00:00") for t in d["date"]]
    data[symbol] = d
  return data
