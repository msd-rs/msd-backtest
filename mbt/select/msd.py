import time

from mbt.shared import get_limit_ratio
import pymsd
from .selector import SelectorDataProvider
import polars as pl
import alpha as al
import pymsd
import logging
import numpy as np

logger = logging.getLogger("selector")



def symbols_where(symbols: list[str]) -> str:
  return ",".join( map(lambda o: f'"{o}"', symbols) )

def db_fields(fields: list[str]) -> str:
  return ",".join(fields)

def to_pl(table: pymsd.Table) -> pl.DataFrame:
  return pl.DataFrame(table.to_dict())

class MsdSelectorDataProvider(SelectorDataProvider):

  def __init__(self, msd_host: str):
    super().__init__()
    self._msd_host = msd_host
    self._client = pymsd.MsdConnection(msd_host)


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
    
    logger.debug(f"start load_concat kline for {len(symbols)} symbols, lastN={lastN}")
    t1 = time.time()
    where_ = symbols_where(symbols)
    sql = f"""
SELECT * FROM stock_kline_1d WHERE obj in ({where_}) limit -{start[0]};
SELECT * FROM stock_dividend WHERE obj in ({where_}) limit -{start[1]};
SELECT * FROM stock_shares WHERE obj in ({where_}) limit -{start[2]};
"""
    table, symbols, bars = self._client.query_join(sql, 'stock_kline_1d', {"stock_dividend": "zero", "stock_shares": "backward"}, symbols[0])

    data = dict((k, v) for k, v in table.to_list())

    logger.info(f"raw data build, {time.time() - t1:.3f}s")

    
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

    dfs = {}
    bars = len(data["close"]) // len(symbols)
    full_df = pl.DataFrame(data)
    for i, symbol in enumerate(symbols):
      dfs[symbol] = full_df[i*bars: (i+1)*bars]
    return dfs
    
  def load_financial(self, symbols: list[str], fields: list[str], only_year: bool = True, lastN: int = 3) -> dict[str, pl.DataFrame]:
    
    logger.debug(f"start load financial for {len(symbols)} symbols, lastN={lastN}")
    year_where = "AND ts.month=12" if only_year else ""
    sql = f"SELECT {db_fields(fields)}  FROM stock_financial where obj in ({symbols_where(symbols)}) {year_where} limit -{lastN}"
    dfs = self._client.query(sql)
    logger.debug(f"finish load financial for {len(symbols)} symbols")

    data = {}
    for symbol, v in dfs.items():
      df = pl.DataFrame( to_pl(v['stock_financial']) )
      data[symbol] = df
    return data


  def load_snapshot(self, symbols: list[str], fin_fields: list[str] = [], fin_only_year: bool = True) -> pl.DataFrame:

    table = self._client.snapshot([
      "stock_kline_1d",
      "stock_shares",
    ], symbols)

    df = to_pl(table)

    if len(fin_fields) > 0:

      rows = []
      objs = []
      for obj, fin_df in self.load_financial(symbols, fin_fields, fin_only_year, 1).items():
        if len(fin_df) > 1:
          fin_df = fin_df.tail(1)
        fin_df = fin_df.rename({"ts": "f_ts"})
        rows.append(fin_df)
        objs.append(obj)
      fin_df = pl.concat(rows).insert_column(0, pl.Series("obj", objs))
      return pl.concat([df, fin_df],  how='align_left')

    return df



if __name__ == "__main__":
  import os
  import logging
  from mbt.shared import A_STOCKS_EXCLUDE_ST
  logging.basicConfig(level=logging.INFO)
  msd_host = os.environ.get("MSD_HOST", "http://localhost:50511") 
  if not msd_host:
    raise Exception("MSD_HOST is not set")
  dp = MsdSelectorDataProvider(msd_host)
  klines = dp.load_kline(["SH600000"], 200)
  financial = dp.load_financial(["SH600000"], ["f001", "f007"], False, 12)
  print(klines)
  print(financial)

  snap = dp.load_snapshot(["SH600000", "SZ002828"], ["f001", "f007"])
  print(snap)
  