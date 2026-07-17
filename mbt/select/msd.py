from encodings import undefined
from encodings import undefined
from .selector import SelectorDataProvider
import pandas as pd
import alpha as al
import pymsd


class MsdSelectorDataProvider(SelectorDataProvider):

  def __init__(self, msd_host: str):
    super().__init__()
    self._msd_host = msd_host
    self._client = pymsd.create_msd_pandas(msd_host)


  def load_kline(self, symbols: list[str], lastN: int = 100) -> dict[str, pd.DataFrame]:
    dividend_lastN = max(lastN//20, 5)
    shares_lastN = max(lastN//10, 5)
    dfs = self._client.load(
      objs=symbols,
      tables=["stock_kline_1d", "stock_dividend", "stock_shares"],
      join={"stock_dividend": "zero", "*": "backward"},
      start=[lastN, dividend_lastN, shares_lastN],
      end=None,
    )

    
    # Apply forward adjustment factor to price related columns
    al.set_ctx(groups=1, flags=al.FLAG_SKIP_NAN)
    for _, data in dfs.items():
      dividends = data["dividend"].to_numpy()
      transfer_shares = data["transfer_shares"].to_numpy()
      right_shares = data["right_shares"].to_numpy()
      right_price = data["right_price"].to_numpy()
      data["close"] = al.FW_SPLIT(data["close"].to_numpy(), dividends, transfer_shares, right_shares, right_price)
      data["open"] = al.FW_SPLIT(data["open"].to_numpy(), dividends, transfer_shares, right_shares, right_price)
      data["high"] = al.FW_SPLIT(data["high"].to_numpy(), dividends, transfer_shares, right_shares, right_price)
      data["low"] = al.FW_SPLIT(data["low"].to_numpy(), dividends, transfer_shares, right_shares, right_price)

    return dfs
    
  def load_financial(self, symbols: list[str], fields: list[str], only_year: bool = True, lastN: int = 3) -> dict[str, pd.DataFrame]:
    if only_year:
      lastN = (lastN + 1) * 4
    dfs = self._client.load(
      objs=symbols,
      tables=["stock_financial"],
      start=lastN,
      end=None,
      fields=fields,
    )

    data = {}
    for symbol, v in dfs.items():
      df = v['stock_financial']
      if only_year:
        data[symbol] = df[df['ts'].dt.month == 12]
      else:
        data[symbol] = df
    return data
    

if __name__ == "__main__":
  import os
  import logging
  logging.basicConfig(level=logging.DEBUG)
  msd_host = os.environ.get("MSD_HOST", "http://localhost:50511") 
  if not msd_host:
    raise Exception("MSD_HOST is not set")
  dp = MsdSelectorDataProvider(msd_host)
  klines = dp.load_kline(["SH600000"], 200)
  financial = dp.load_financial(["SH600000"], ["f001", "f007"], 12)
  #print(klines)
  print(financial)
  print(dp.snapshot_last([klines, financial]))