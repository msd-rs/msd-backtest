# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from datetime import datetime, date
import itertools
import time
from typing import Tuple

from mbt.shared import get_limit_ratio, to_statutory_disclosure_date
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
  """Concrete SelectorDataProvider implementation powered by MSD database service.

  Fetches stock klines, financial reports, snapshots, and corporate actions
  from an MSD service endpoint, performing dividend adjustments and statutory
  disclosure alignment.
  """

  def __init__(self, msd_host: str, date: str | None = None):
    super().__init__()
    self.date = date
    self._msd_host = msd_host
    self._client = pymsd.MsdConnection(msd_host)

  def load_kline(self, objs: list[str], lastN: int | list[int] = 100,
                 fin_fields: list[str] = [], fin_only_year: bool = True,
                 baseline: str | None = None) -> Panel:
    """Load klines with corporate action adjustments and financial metrics.

    See SelectorDataProvider.load_kline for layout contract and column semantics.
    """
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

    t1 = time.time()
    table, loaded, bars = self._client.query_join(
      sql,
      'stock_kline_1d',
      {"stock_dividend": "zero", "stock_shares": "backward"},
      baseline_sym,
    )
    data = table.to_dict()
    logger.info(f"raw data build, {time.time() - t1:.3f}s, {len(loaded)} objs x {bars} bars")
    if bars < start[0]:
      logger.warning(
        f"load_kline requested {start[0]} bars, but only {bars} bars available (date={self.date})"
      )

    # Align financial fields with statutory disclosure dates (strictly no lookahead)
    if len(fin_fields) > 0 and len(loaded) > 0 and bars > 0:
      ts_year_where = "AND ts.month=12" if fin_only_year else ""
      n = max(start[0] // 100 + 4, 4) if fin_only_year else max(start[0] // 50 + 4, 8)
      fin_query_cols = list(fin_fields)
      if "ts" not in fin_query_cols:
        fin_query_cols.insert(0, "ts")
      fin_sql = f"SELECT {','.join(fin_query_cols)} FROM stock_financial WHERE obj in ({where_}) {date_where} {ts_year_where} LIMIT -{n};"
      for col in fin_fields:
        data[col] = np.full(len(loaded) * bars, np.nan, dtype=np.float64)
      data["report_ts"] = np.full(len(loaded) * bars, np.datetime64("NaT", "us"), dtype="datetime64[us]")

      try:
        res = self._client.query_concat(fin_sql)
        if "stock_financial" in res:
          table_fin, ranges = res["stock_financial"]
          d_fin = table_fin.to_dict()
          if len(d_fin.get("ts", [])) > 0:
            rep_ts_raw = d_fin["ts"]
            ann_ts_raw = to_statutory_disclosure_date(pl.Series("report_ts", rep_ts_raw)).to_numpy()
            range_map = {r[0]: (r[1], r[2]) for r in ranges}
            for g, obj in enumerate(loaded):
              if obj not in range_map:
                continue
              st, end_idx = range_map[obj]
              if st >= end_idx:
                continue
              obj_ann_ts = ann_ts_raw[st:end_idx]
              if len(obj_ann_ts) == 0:
                continue
              obj_rep_ts = rep_ts_raw[st:end_idx]
              kline_slice = slice(g * bars, (g + 1) * bars)
              kline_ts = data["ts"][kline_slice]

              idx = np.searchsorted(obj_ann_ts, kline_ts, side="right") - 1
              valid = idx >= 0
              safe_idx = np.maximum(idx, 0)

              for col in fin_fields:
                if col in d_fin:
                  val = d_fin[col][st:end_idx]
                  if len(val) == 0:
                    continue
                  col_arr = np.full(bars, np.nan, dtype=np.float64)
                  col_arr[valid] = val[safe_idx[valid]]
                  data[col][kline_slice] = col_arr

              rep_arr = np.full(bars, np.datetime64("NaT", "us"), dtype="datetime64[us]")
              rep_arr[valid] = obj_rep_ts[safe_idx[valid]]
              data["report_ts"][kline_slice] = rep_arr
      except Exception as e:
        logger.error(f"failed to query and align stock_financial in load_kline: {e}", exc_info=True)
        raise

    # Ensure corporate action columns exist even if no dividend events occurred
    for col in ["dividend", "transfer_shares", "right_shares", "right_price"]:
      if col not in data:
        data[col] = np.zeros_like(data["close"])

    # groups 必须等于返回的股票数(而非入参数量: 可能重复/未排序)
    # 使用 al.Context 进行短周期上下文管理，避免污染全局上下文
    with al.Context(groups=len(loaded), flags=al.FLAG_SKIP_NAN):
      # limit_ratio 必须按返回块顺序生成, 且基于未复权除权参考价计算涨跌停, 避免前复权缩放后 round 失真
      limit_ratio = get_limit_ratio(loaded, repeat=bars)
      prev_raw_close = al.REF(data["close"], 1)
      denom = 1.0 + data["transfer_shares"] + data["right_shares"]
      denom = np.where(denom > 0, denom, 1.0)
      ex_ref_price = (
        prev_raw_close - data["dividend"] + data["right_shares"] * data["right_price"]
      ) / denom
      limit_up_price = np.round(ex_ref_price * (1.0 + limit_ratio), 2)
      limit_down_price = np.round(ex_ref_price * (1.0 - limit_ratio), 2)
      valid_ref = ~np.isnan(prev_raw_close) & (prev_raw_close > 0)
      is_limit_up = valid_ref & (data["close"] == data["high"]) & (data["close"] >= limit_up_price)
      is_limit_down = valid_ref & (data["close"] == data["low"]) & (data["close"] <= limit_down_price)
      data["limited"] = np.where(is_limit_up, 1, np.where(is_limit_down, -1, 0))

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

  def load_financial(self, objs: list[str], fields: list[str], only_year: bool = True, lastN: int = 12) -> pl.DataFrame:
    """Load historical financial records aligned by statutory disclosure deadline.

    See SelectorDataProvider.load_financial for schema details.
    """
    if not objs or not fields:
      return pl.DataFrame()

    logger.debug(f"start load financial objs, lastN={lastN}")
    # Always include ts in query to derive statutory announcement date
    query_fields = list(fields)
    if "ts" not in query_fields:
      query_fields.insert(0, "ts")

    conds = []
    fetch_n = int(lastN)
    if self.date:
      conds.append(f'ts <= "{self.date}"')
      try:
        d_str = str(self.date)[:10]
        dt_part = datetime.fromisoformat(d_str)
        md = (dt_part.month, dt_part.day)
        # During statutory disclosure lag windows (Jan 1-Apr 29, Jun 30-Aug 30, Sep 30-Oct 30),
        # the latest report_ts <= date may not be disclosed yet; fetch extra rows so post-filter retains lastN
        if (1, 1) <= md < (4, 30) or (6, 30) <= md < (8, 31) or (9, 30) <= md < (10, 31):
          fetch_n = lastN + (1 if only_year else 2)
      except Exception:
        pass
    if only_year:
      conds.append("ts.month = 12")
    date_where = f"AND {' AND '.join(conds)}" if conds else ""
    sql = f"SELECT {db_fields(query_fields)} FROM stock_financial where obj in ({objs_where(objs)}) {date_where} limit -{fetch_n}"
    res = self._client.query_concat(sql)

    if "stock_financial" not in res:
      logger.warning("no stock_financial in result")
      return pl.DataFrame()
    table, ranges = res["stock_financial"]
    logger.debug(f"finish load financial for {len(ranges)} objs")

    total_rows = table.rows
    if total_rows > 0:
      fin_objs = np.repeat([r[0] for r in ranges], [r[2] - r[1] for r in ranges])
    else:
      fin_objs = np.array([], dtype=str)

    d = table.to_dict()
    rep_ts = d.get("ts")
    if rep_ts is None or len(rep_ts) == 0:
      return pl.DataFrame()

    ann_ts = to_statutory_disclosure_date(pl.Series("report_ts", rep_ts))

    df_data = {
      "obj": fin_objs,
      "ts": ann_ts,
      "report_ts": rep_ts,
    }
    for f in fields:
      if f in d:
        df_data[f] = d[f]

    df = pl.DataFrame(df_data)

    if self.date:
      if isinstance(self.date, str):
        target_dt = datetime.fromisoformat(self.date)
      elif isinstance(self.date, datetime):
        target_dt = self.date
      elif isinstance(self.date, date):
        target_dt = datetime(self.date.year, self.date.month, self.date.day)
      else:
        target_dt = pl.lit(self.date).str.to_datetime(time_unit="us")
      df = df.filter(pl.col("ts") <= target_dt)

    if only_year:
      df = df.filter(pl.col("report_ts").dt.month() == 12)

    if len(df) == 0:
      return pl.DataFrame()

    df = (
      df.sort(["obj", "report_ts"])
      .group_by("obj", maintain_order=True)
      .tail(lastN)
    )
    logger.debug(f"finish data frame financial for {len(ranges)} objs")
    return df


  def load_snapshot(self, objs: list[str], fin_fields: list[str] = [], fin_only_year: bool = True) -> pl.DataFrame:
    """Take a cross-sectional snapshot for screening at the latest trading day.

    See SelectorDataProvider.load_snapshot for return schema details.
    """
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
