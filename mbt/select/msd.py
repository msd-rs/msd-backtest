import itertools
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


def objs_where(objs: list[str]) -> str:
  return ",".join(map(lambda o: f'"{o}"', objs))


def db_fields(fields: list[str]) -> str:
  return ",".join(fields)


def to_pl(table: pymsd.Table) -> pl.DataFrame:
  return pl.DataFrame(table.to_dict())


class MsdSelectorDataProvider(SelectorDataProvider):

  def __init__(self, msd_host: str, date: str | None = None):
    super().__init__()
    self.date = date
    self._msd_host = msd_host
    self._client = pymsd.MsdConnection(msd_host)

  def load_kline(self, objs: list[str], lastN: int | list[int] = 100,
                 fin_fields: list[str] = [], fin_only_year: bool = True,
                 baseline: str | None = None) -> Panel:
    """见 SelectorDataProvider.load_kline 的布局契约与列语义说明"""
    if len(objs) == 0 and baseline is None:
      raise ValueError("objs is empty and no baseline given")

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
    request = sorted(set(objs))
    if baseline is not None and baseline not in request:
      request = sorted({*request, baseline})
    baseline_sym = baseline if baseline is not None else request[0]

    where_ = objs_where(request)
    date_where = f'AND ts <= "{self.date}"' if self.date else ""
    sql = f"""
SELECT * FROM stock_kline_1d WHERE obj in ({where_}) {date_where} limit -{start[0]};
SELECT * FROM stock_dividend WHERE obj in ({where_}) {date_where} limit -{start[1]};
SELECT * FROM stock_shares WHERE obj in ({where_}) {date_where} limit -{start[2]};
"""
    if len(fin_fields) > 0:
      ts_year_where = "AND ts.month=12" if fin_only_year else ""
      n = max(start[0] // 100, 1) if fin_only_year else max(start[0] // 50, 4)
      sql += f"SELECT {','.join(fin_fields)} FROM stock_financial WHERE obj in ({where_}) {date_where} {ts_year_where} LIMIT  -{n};\n"

    t1 = time.time()
    table, loaded, bars = self._client.query_join(
      sql,
      'stock_kline_1d',
      {"stock_dividend": "zero", "stock_shares": "backward", "stock_financial": "backward"},
      baseline_sym,
    )
    data = table.to_dict()
    logger.info(f"raw data build, {time.time() - t1:.3f}s, {len(loaded)} objs x {bars} bars")
    if bars < start[0]:
      logger.warning(
        f"load_kline requested {start[0]} bars, but only {bars} bars available (date={self.date})"
      )

    # groups 必须等于返回的股票数(而非入参数量: 可能重复/未排序)
    # 使用 al.Context 进行短周期上下文管理，避免污染全局上下文
    with al.Context(groups=len(loaded), flags=al.FLAG_SKIP_NAN):
      # limit_ratio 必须按返回块顺序生成, 且基于未复权除权参考价计算涨跌停, 避免前复权缩放后 round 失真
      limit_ratio = get_limit_ratio(loaded, repeat=bars)
      prev_raw_close = np.nan_to_num(al.REF(data["close"], 1))
      denom = 1.0 + data["transfer_shares"] + data["right_shares"]
      denom = np.where(denom > 0, denom, 1.0)
      ex_ref_price = (
        prev_raw_close - data["dividend"] + data["right_shares"] * data["right_price"]
      ) / denom
      data["limited"] = np.where(
        (data["close"] >= np.round(ex_ref_price * (1 + limit_ratio), 2)), 1, np.where(
        (data["close"] <= np.round(ex_ref_price * (1 - limit_ratio), 2)), -1, 0))

      # Preserve unadjusted raw prices before forward adjustment
      data["raw_open"] = data["open"].copy()
      data["raw_high"] = data["high"].copy()
      data["raw_low"] = data["low"].copy()
      data["raw_close"] = data["close"].copy()
      data["price"] = data["close"].copy()
      if "volume" in data and "amount" in data:
        with np.errstate(divide="ignore", invalid="ignore"):
          data["raw_vwap"] = np.where(data["volume"] > 0, data["amount"] / data["volume"], np.nan)

      # Apply forward adjustment factor to price related columns
      data["open"] = al.FW_SPLIT(data["open"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
      data["high"] = al.FW_SPLIT(data["high"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
      data["low"] = al.FW_SPLIT(data["low"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
      data["close"] = al.FW_SPLIT(data["close"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])

    full_df = pl.DataFrame(data)

    # 剔除对齐基准(通常是指数): 它的块只用于定义时间轴, 不属于选股结果
    if baseline is not None and baseline in loaded:
      idx = loaded.index(baseline)
      keep = np.ones(len(loaded), dtype=bool)
      keep[idx] = False
      full_df = full_df.filter(pl.Series("keep", np.repeat(keep, bars)))
      loaded = [s for s in loaded if s != baseline]

    if bars <= 0:
      raise ValueError(f"no bars loaded for {len(objs)} objs")

    return Panel(full_df, loaded, bars)

  def load_financial(self, objs: list[str], fields: list[str], only_year: bool = True, lastN: int = 3) -> pl.DataFrame:
    """见 SelectorDataProvider.load_financial"""
    if not objs:
      return pl.DataFrame()

    logger.debug(f"start load financial objs, lastN={lastN}")
    date_where = f'AND ts <= "{self.date}"' if self.date else ""
    year_where = "AND ts.month=12" if only_year else ""
    sql = f"SELECT {db_fields(fields)}  FROM stock_financial where obj in ({objs_where(objs)}) {date_where} {year_where} limit -{lastN}"
    res = self._client.query_concat(sql)

    if "stock_financial" not in res:
      logger.warning("no stock_financial in result")
      return pl.DataFrame()
    table, ranges = res["stock_financial"]
    logger.debug(f"finish load financial for {len(ranges)} objs")

    total_rows = table.rows
    if total_rows > 0:
      objs = np.repeat([r[0] for r in ranges], [r[2] - r[1] for r in ranges])
    else:
      objs = np.array([], dtype=str)

    cols = [pl.Series("obj", objs, dtype=pl.String)]
    d = table.to_dict()
    for k, v in d.items():
      cols.append(pl.Series(k, v))

    df = pl.DataFrame(cols)
    logger.debug(f"finish data frame financial for {len(ranges)} objs")
    return df


  def load_snapshot(self, objs: list[str], fin_fields: list[str] = [], fin_only_year: bool = True) -> pl.DataFrame:
    """见 SelectorDataProvider.load_snapshot: 每只股票一行, 直接可做截面筛选"""
    table = self._client.snapshot([
      "stock_kline_1d",
      "stock_shares",
    ], objs, self.date)

    df = to_pl(table)
    raw_cols = []
    for col in ["open", "high", "low", "close"]:
      if col in df.columns and f"raw_{col}" not in df.columns:
        raw_cols.append(pl.col(col).alias(f"raw_{col}"))
    if "amount" in df.columns and "volume" in df.columns and "raw_vwap" not in df.columns:
      raw_cols.append(
        pl.when(pl.col("volume") > 0)
        .then(pl.col("amount") / pl.col("volume"))
        .otherwise(None)
        .alias("raw_vwap")
      )
    if raw_cols:
      df = df.with_columns(raw_cols)

    if len(fin_fields) > 0:
      fin_df = self.load_financial(objs, fin_fields, fin_only_year, 1)
      if len(fin_df) > 0:
        if len(fin_df) > 1:
          fin_df = fin_df.group_by("obj", maintain_order=True).last()
        if "ts" in fin_df.columns:
          fin_df = fin_df.rename({"ts": "f_ts"})
        return df.join(fin_df, on="obj", how="left")

    return df


if __name__ == "__main__":
  import os
  import logging
  logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.DEBUG
)
  msd_host = os.environ.get("MSD_HOST", "http://localhost:50511")
  if not msd_host:
    raise Exception("MSD_HOST is not set")
  dp = MsdSelectorDataProvider(msd_host)
  table = dp.load_financial(['*'], ["f001", "f007"], False, 12)
  print(table)
