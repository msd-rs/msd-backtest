from mbt.shared import get_limit_ratio
import pymsd
from .selector import SelectorDataProvider
import polars as pl
import alpha as al
import pymsd
import logging
import numpy as np

logger = logging.getLogger("selector")


class MsdSelectorDataProvider(SelectorDataProvider):

  def __init__(self, msd_host: str):
    super().__init__()
    self._msd_host = msd_host
    self._client = pymsd.create_msd_polars(msd_host)


  def load_kline(self, symbols: list[str], lastN: int | list[int] = 100) -> dict[str, pl.DataFrame]:

    start : list[int] = []
    if isinstance(lastN, int):
      if lastN == 1:
        dividend_lastN = 1
        shares_lastN = 1
      else:
        dividend_lastN = max(lastN//20, 5)
        shares_lastN = max(lastN//10, 5)
      start = [lastN, dividend_lastN, shares_lastN]
    else:
      if not isinstance(lastN, list) or len(lastN) != 3:
        raise ValueError("lastN must be a list of 3 integers")
      start = lastN
    

    logger.debug(f"start load kline for {len(symbols)} symbols, lastN={lastN}")
    # pyrefly: ignore [no-matching-overload]
    dfs = self._client.load(
      objs=symbols,
      tables=["stock_kline_1d", "stock_dividend", "stock_shares"],
      join={"stock_dividend": "zero", "*": "backward"},
      start=start,
      end=self.date,
    )
    logger.debug(f"finish load kline for {len(symbols)} symbols")


    data, symbols = self._client.adaptor.concat(dfs, base=symbols[0], join="backward")
    logger.info("data concatenated")
    if data is None or "ts" not in data:
      return {} 

    data = self._client.adaptor.to_numpy(data)
    for col in ["dividend", "transfer_shares", "right_shares", "right_price"]:
      if col not in data:
        data[col] = np.zeros_like(data["close"])

    
    # Apply forward adjustment factor to price related columns
    al.set_ctx(groups=len(symbols),  flags=al.FLAG_SKIP_NAN)

    data["open"] =  al.FW_SPLIT(data["open"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
    data["high"] =  al.FW_SPLIT(data["high"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
    data["low"] =  al.FW_SPLIT(data["low"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
    data["close"] =  al.FW_SPLIT(data["close"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])

    limit_ratio = get_limit_ratio(symbols, repeat=len(data["close"])//len(symbols))
    last_close = np.nan_to_num(al.REF(data["close"], 1))
    data["limited"] = np.where(
      (data["close"] >= np.round(last_close * (1 + limit_ratio), 2)), 1, np.where(
      (data["close"] <= np.round(last_close * (1 - limit_ratio), 2)), -1, 0))

    # for symbol, data in dfs.items():
    #   if len(data["ts"]) == 1:
    #     # Skip if there is only one day of data
    #     continue
    #   dividends = data["dividend"].to_numpy()
    #   transfer_shares = data["transfer_shares"].to_numpy()
    #   right_shares = data["right_shares"].to_numpy()
    #   right_price = data["right_price"].to_numpy()
    #   close = al.FW_SPLIT(data["close"].to_numpy(), dividends, transfer_shares, right_shares, right_price)

    #   last_close = np.nan_to_num(al.REF(close, 1), False)
    #   limit_ratio = get_limit_ratio([symbol], 1)
    #   limited = np.where(
    #     (close >= np.round(last_close * (1 + limit_ratio), 2)), 1, np.where(
    #       (close <= np.round(last_close * (1 - limit_ratio), 2)), -1, 0))

    #   dfs[symbol] = data.with_columns(
    #     close=close,
    #     open=al.FW_SPLIT(data["open"].to_numpy(), dividends, transfer_shares, right_shares, right_price),
    #     high=al.FW_SPLIT(data["high"].to_numpy(), dividends, transfer_shares, right_shares, right_price),
    #     low=al.FW_SPLIT(data["low"].to_numpy(), dividends, transfer_shares, right_shares, right_price),
    #     limited = limited
    #   )
    # logger.debug(f"finish build kline for {len(symbols)} symbols")

    dfs = {}
    bars = len(data["close"]) // len(symbols)
    full_df = pl.DataFrame(data)
    for i, symbol in enumerate(symbols):
      dfs[symbol] = full_df[i*bars: (i+1)*bars]
    return dfs
    
  def load_financial(self, symbols: list[str], fields: list[str], only_year: bool = True, lastN: int = 3) -> dict[str, pl.DataFrame]:
    if only_year:
      lastN = lastN * 4 + 1
    
    logger.debug(f"start load financial for {len(symbols)} symbols, lastN={lastN}")
    dfs = self._client.load(
      objs=symbols,
      tables=["stock_financial"],
      start=lastN,
      end=self.date,
      fields=fields,
    )
    logger.debug(f"finish load financial for {len(symbols)} symbols")

    data = {}
    for symbol, v in dfs.items():
      df = v['stock_financial']
      # df = df.with_columns(
      #   pl.col("ts").map_elements(lambda d: int(d.year), return_dtype=pl.Int32).alias("f_year"),
      #   pl.col("ts").map_elements(lambda d: int(d.month/3), return_dtype=pl.Int32).alias("f_quarter")
      # )
      if only_year:
        data[symbol] = df.filter(pl.col('ts').dt.month() == 12)
      else:
        data[symbol] = df
    return data


  def load_snapshot(self, symbols: list[str], fin_fields: list[str] = [], fin_only_year: bool = True) -> pl.DataFrame:

    dfs = self.load_kline(symbols, [2, 1, 1])

    dfs = {k: df.tail(1) for k, df in dfs.items()}

    if len(fin_fields) > 0:
      for obj, fin_df in self.load_financial(symbols, fin_fields, fin_only_year, 1).items():
        if obj in dfs:
          if len(fin_df) > 1:
            fin_df = fin_df.tail(1)
          dfs[obj] = pl.concat([dfs[obj], fin_df.rename({"ts": "f_ts"})], how='horizontal_extend')


    rows = [
      df.with_columns(
        pl.lit(obj).alias('obj')
      ) for obj, df in dfs.items()
    ]

    return pl.concat(rows)


if __name__ == "__main__":
  import os
  import logging
  from mbt.shared import A_STOCKS_EXCLUDE_ST
  logging.basicConfig(level=logging.INFO)
  msd_host = os.environ.get("MSD_HOST", "http://localhost:50511") 
  if not msd_host:
    raise Exception("MSD_HOST is not set")
  dp = MsdSelectorDataProvider(msd_host)
  # klines = dp.load_kline(["SH600000"], 200)
  # financial = dp.load_financial(["SH600000"], ["f001", "f007"], False, 12)
  # #print(klines)
  # print(financial)

  snap = dp.load_snapshot(["SH600000", "SZ002828"])
  print(snap)
  