from .selector import SelectorDataProvider
import polars as pl
import alpha as al
import pymsd
import logging

logger = logging.getLogger("selector")


class MsdSelectorDataProvider(SelectorDataProvider):

  def __init__(self, msd_host: str):
    super().__init__()
    self._msd_host = msd_host
    self._client = pymsd.create_msd_polars(msd_host)


  def load_kline(self, symbols: list[str], lastN: int = 100) -> dict[str, pl.DataFrame]:
    if lastN == 1:
      dividend_lastN = 1
      shares_lastN = 1
    else:
      dividend_lastN = max(lastN//20, 5)
      shares_lastN = max(lastN//10, 5)

    logger.debug(f"start load kline for {len(symbols)} symbols, lastN={lastN}")
    dfs = self._client.load(
      objs=symbols,
      tables=["stock_kline_1d", "stock_dividend", "stock_shares"],
      join={"stock_dividend": "zero", "*": "backward"},
      start=[lastN, dividend_lastN, shares_lastN],
      end=None,
    )
    logger.debug(f"finish load kline for {len(symbols)} symbols")

    
    # Apply forward adjustment factor to price related columns
    al.set_ctx(groups=1, flags=al.FLAG_SKIP_NAN)
    for symbol, data in dfs.items():
      if len(data["ts"]) == 1:
        # Skip if there is only one day of data
        continue
      dividends = data["dividend"].to_numpy()
      transfer_shares = data["transfer_shares"].to_numpy()
      right_shares = data["right_shares"].to_numpy()
      right_price = data["right_price"].to_numpy()
      dfs[symbol] = data.with_columns(
        close=al.FW_SPLIT(data["close"].to_numpy(), dividends, transfer_shares, right_shares, right_price),
        open=al.FW_SPLIT(data["open"].to_numpy(), dividends, transfer_shares, right_shares, right_price),
        high=al.FW_SPLIT(data["high"].to_numpy(), dividends, transfer_shares, right_shares, right_price),
        low=al.FW_SPLIT(data["low"].to_numpy(), dividends, transfer_shares, right_shares, right_price)
      )

    return dfs
    
  def load_financial(self, symbols: list[str], fields: list[str], only_year: bool = True, lastN: int = 3) -> dict[str, pl.DataFrame]:
    if only_year:
      lastN = (lastN + 1) * 4
    
    logger.debug(f"start load financial for {len(symbols)} symbols, lastN={lastN}")
    dfs = self._client.load(
      objs=symbols,
      tables=["stock_financial"],
      start=lastN,
      end=None,
      fields=fields,
    )
    logger.debug(f"finish load financial for {len(symbols)} symbols")

    data = {}
    for symbol, v in dfs.items():
      df = v['stock_financial']
      if only_year:
        data[symbol] = df.filter(pl.col('ts').dt.month() == 12)
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