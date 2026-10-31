import time
from typing import Tuple

from mbt.shared import get_limit_ratio
import pymsd
from mbt.select.panel import Panel
from mbt.select.selector import SelectorDataProvider
import polars as pl
import alpha as al
import logging
import numpy as np

logger = logging.getLogger("selector")


def symbols_where(symbols: list[str]) -> str:
  return ",".join(map(lambda o: f'"{o}"', symbols))


def db_fields(fields: list[str]) -> str:
  return ",".join(fields)


def to_pl(table: pymsd.Table) -> pl.DataFrame:
  return pl.DataFrame(table.to_dict())


class MsdSelectorDataProvider(SelectorDataProvider):

  def __init__(self, msd_host: str):
    super().__init__()
    self._msd_host = msd_host
    self._client = pymsd.MsdConnection(msd_host)

  def load_kline(self, symbols: list[str], lastN: int | list[int] = 100,
                 fin_fields: list[str] = [], fin_only_year: bool = True,
                 baseline: str | None = None) -> Panel:
    """见 SelectorDataProvider.load_kline 的布局契约与列语义说明"""
    if len(symbols) == 0 and baseline is None:
      raise ValueError("symbols is empty and no baseline given")

    # lastN: int 或 [kline_bars, dividend_events, shares_events]
    if isinstance(lastN, int):
      if lastN == 1:
        start = [1, 1, 1]
      else:
        start = [lastN, max(lastN // 50, 2), max(lastN // 50, 2)]
    else:
      if not isinstance(lastN, (list, tuple)) or len(lastN) != 3:
        raise ValueError("lastN must be an int or a list of 3 ints")
      start = [int(x) for x in lastN]

    # 排序去重, 让返回行序可复现; baseline 自动并入请求并保证成为对齐基准
    request = sorted(set(symbols))
    if baseline is not None and baseline not in request:
      request = sorted({*request, baseline})
    baseline_sym = baseline if baseline is not None else request[0]

    where_ = symbols_where(request)
    sql = f"""
SELECT * FROM stock_kline_1d WHERE obj in ({where_}) limit -{start[0]};
SELECT * FROM stock_dividend WHERE obj in ({where_}) limit -{start[1]};
SELECT * FROM stock_shares WHERE obj in ({where_}) limit -{start[2]};
"""
    if len(fin_fields) > 0:
      ts_year_where = "AND ts.month=12" if fin_only_year else ""
      n = max(start[0] // 100, 1) if fin_only_year else max(start[0] // 50, 4)
      sql += f"SELECT {','.join(fin_fields)} FROM stock_financial WHERE obj in ({where_}) {ts_year_where} LIMIT  -{n};\n"

    t1 = time.time()
    table, loaded, bars = self._client.query_join(
      sql,
      'stock_kline_1d',
      {"stock_dividend": "zero", "stock_shares": "backward", "stock_financial": "backward"},
      baseline_sym,
    )
    data = table.to_dict()
    logger.info(f"raw data build, {time.time() - t1:.3f}s, {len(loaded)} symbols x {bars} bars")

    # groups 必须等于返回的股票数(而非入参数量: 可能重复/未排序)
    al.set_ctx(groups=len(loaded), flags=al.FLAG_SKIP_NAN)

    # Apply forward adjustment factor to price related columns
    data["open"] = al.FW_SPLIT(data["open"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
    data["high"] = al.FW_SPLIT(data["high"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
    data["low"] = al.FW_SPLIT(data["low"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
    data["close"] = al.FW_SPLIT(data["close"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])

    # limit_ratio 必须按返回块顺序生成, 否则涨跌停标记会错配到别的股票
    limit_ratio = get_limit_ratio(loaded, repeat=bars)
    last_close = np.nan_to_num(al.REF(data["close"], 1))
    data["limited"] = np.where(
      (data["close"] >= np.round(last_close * (1 + limit_ratio), 2)), 1, np.where(
      (data["close"] <= np.round(last_close * (1 - limit_ratio), 2)), -1, 0))

    full_df = pl.DataFrame(data)

    # 剔除对齐基准(通常是指数): 它的块只用于定义时间轴, 不属于选股结果
    if baseline is not None and baseline in loaded:
      idx = loaded.index(baseline)
      keep = np.ones(len(loaded), dtype=bool)
      keep[idx] = False
      full_df = full_df.filter(pl.Series("keep", np.repeat(keep, bars)))
      loaded = [s for s in loaded if s != baseline]

    if bars <= 0:
      raise ValueError(f"no bars loaded for {len(symbols)} symbols")

    return Panel(full_df, loaded, bars)

  def load_financial(self, symbols: list[str], fields: list[str], only_year: bool = True, lastN: int = 3) -> dict[str, pl.DataFrame]:
    """见 SelectorDataProvider.load_financial"""
    logger.debug(f"start load financial for {len(symbols)} symbols, lastN={lastN}")
    year_where = "AND ts.month=12" if only_year else ""
    sql = f"SELECT {db_fields(fields)}  FROM stock_financial where obj in ({symbols_where(symbols)}) {year_where} limit -{lastN}"
    dfs = self._client.query(sql)
    logger.debug(f"finish load financial for {len(symbols)} symbols")

    data = {}
    for symbol, v in dfs.items():
      df = pl.DataFrame(to_pl(v['stock_financial']))
      data[symbol] = df
    return data

  def load_snapshot(self, symbols: list[str], fin_fields: list[str] = [], fin_only_year: bool = True) -> pl.DataFrame:
    """见 SelectorDataProvider.load_snapshot: 每只股票一行, 直接可做截面筛选"""
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
      return pl.concat([df, fin_df], how='align_left')

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
  panel = dp.load_kline(["SH600000", "SH600001"], 200, baseline="SH000001")
  print(panel)
  print(panel.cross_section(close=panel.last("close")))
