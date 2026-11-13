# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

import datetime
import time
import logging
import numpy as np
import polars as pl
import alpha as al
import pymsd
from mbt import DataProvider
from mbt.shared import get_limit_ratio, to_statutory_disclosure_date

logger = logging.getLogger("msd_loader")


def objs_where(objs: list[str]) -> str:
  return ",".join(map(lambda o: f'"{o}"', objs))


def db_fields(fields: list[str] | str) -> str:
  if isinstance(fields, str):
    return fields
  return ",".join(fields)


def next_hook_al(i: int, groups: int, bars: int):
  ctx = al.Context.current()
  ctx.end = i + 1
  ctx.apply()


def find_date_index(
  dates: np.ndarray, target_date: str | datetime.date | datetime.datetime
) -> int:
  """Find the first index where dates >= target_date using binary search.

  Supports np.datetime64, strings, bytes, datetime objects, and numeric timestamps.
  """
  if len(dates) == 0:
    return 0

  target_str = str(target_date)[:10]
  if hasattr(dates, "dtype") and np.issubdtype(dates.dtype, np.datetime64):
    target_val = np.datetime64(target_str).astype(dates.dtype)
    return int(np.searchsorted(dates, target_val))

  first = dates[0]
  if isinstance(first, bytes):
    return int(np.searchsorted(dates, target_str.encode()))

  if isinstance(first, str) or (
    hasattr(dates, "dtype") and np.issubdtype(dates.dtype, np.character)
  ):
    return int(np.searchsorted(dates, target_str))

  if isinstance(first, datetime.datetime):
    dt = datetime.datetime.fromisoformat(target_str)
    return int(np.searchsorted(dates, dt))

  if isinstance(first, datetime.date):
    d = datetime.date.fromisoformat(target_str)
    return int(np.searchsorted(dates, d))

  v0 = float(first)
  dt = datetime.datetime.fromisoformat(target_str)
  if v0 > 1e17:
    target_val = int(dt.timestamp() * 1e9)
  elif v0 > 1e14:
    target_val = int(dt.timestamp() * 1e6)
  elif v0 > 1e11:
    target_val = int(dt.timestamp() * 1000)
  elif v0 > 1e8:
    target_val = int(dt.timestamp())
  else:
    target_val = int(target_str.replace("-", ""))
  return int(np.searchsorted(dates, target_val))


def _check_dataset_min_ts(client: pymsd.MsdConnection, obj: str, req_start: str | int | None):
  if not req_start or isinstance(req_start, int):
    return
  req_start_str = str(req_start)[:10]
  try:
    res = client.query_concat(
      f'SELECT ts FROM stock_kline_1d WHERE obj="{obj}" LIMIT 1; '
      f'SELECT ts FROM stock_kline_1d WHERE obj="{obj}" LIMIT -1;'
    )
    if "stock_kline_1d" in res:
      t, _ = res["stock_kline_1d"]
      d = t.to_dict().get("ts", [])
      if len(d) > 0:
        min_ts_str = str(d[0])[:10]
        max_ts_str = str(d[-1])[:10]
        if req_start_str < min_ts_str:
          raise ValueError(
            f"Requested start '{req_start_str}' is earlier than dataset min_ts '{min_ts_str}' "
            f"(available range for {obj}: {min_ts_str} to {max_ts_str})."
          )
  except ValueError:
    raise
  except Exception:
    pass


def load_data(
  msd_host: str,
  objs: list[str],
  start: str,
  end: str,
  fin_fields: list[str] | str | None = None,
  fin_only_year: bool = False,
  warmup_bars: int = 0,
  fin_lookback_days: int | None = None,
) -> tuple[DataProvider, list[str]]:
  """Create a DataProvider for backtesting from MSD database.

  Lookahead Bias Protection for Financial Data:
  Financial reporting dates (e.g. 12-31 for annual reports) are NOT actual disclosure dates.
  This function automatically converts report period dates to legal statutory announcement
  deadlines (04-30, 08-31, 10-31) and renames the period end date to `report_ts`.
  Market bars are aligned with financial data strictly on or after statutory disclosure deadlines,
  preventing lookahead bias.

  Args:
    msd_host: MSD server host URL.
    objs: List of stock symbols to load.
    start: Backtest start date (e.g. '2024-01-01').
    end: Backtest end date (e.g. '2025-01-01').
    fin_fields: List of financial fields to query from stock_financial (e.g. ['f001', 'f205']).
      If None or empty (default), NO financial data is loaded, avoiding memory explosion.
      To load all fields, pass fin_fields=['*'] (warning: high memory usage).
    fin_only_year: If True, only query annual reports (ts.month=12) from stock_financial.
    warmup_bars: Number of warmup bars loaded prior to `start` date for indicator initialization.
    fin_lookback_days: Explicit lookback window in calendar days for financial data history.
      Defaults to 1200 days for annual reports (~3.3 years) and 500 days for quarterly reports.

  Returns:
    tuple of (DataProvider, loaded_objs).
  """
  if not objs:
    return DataProvider({"ts": np.array([])}, objs=[]), []

  client = pymsd.MsdConnection(msd_host)

  target_start = start
  query_start = start
  if warmup_bars > 0 and start and not isinstance(start, int):
    lookback_days = int(warmup_bars * 1.8) + 15
    try:
      d = datetime.date.fromisoformat(str(start)[:10])
      query_start = (d - datetime.timedelta(days=lookback_days)).isoformat()
    except ValueError:
      query_start = str(start)

  ts_where = []
  fin_ts_where = []
  shares_ts_where = []
  limit = ""
  fin_limit = ""
  if isinstance(start, int):
    limit = f"limit -{start}"
    fin_limit = f"limit -{max(start // 50, 8)}"
  elif query_start:
    ts_where.append(f"ts >= '{query_start}'")
    # stock_shares: lookback 10 years to find the initial share capital without filtering by ts.month=12
    try:
      shares_start = (
        datetime.date.fromisoformat(str(query_start)[:10]) - datetime.timedelta(days=3650)
      ).isoformat()
    except ValueError:
      shares_start = str(query_start)
    shares_ts_where.append(f"ts >= '{shares_start}'")

    if fin_lookback_days is not None:
      fin_lookback = max(1, int(fin_lookback_days))
    else:
      # Default: 1200 days (~3.3 years) for annual reports to ensure 3-4 annual periods are covered;
      # 500 days (~1.4 years) for quarterly reports.
      fin_lookback = 1200 if fin_only_year else 500
    try:
      fin_start = (
        datetime.date.fromisoformat(str(query_start)[:10]) - datetime.timedelta(days=fin_lookback)
      ).isoformat()
    except ValueError:
      fin_start = str(query_start)
    fin_ts_where.append(f"ts >= '{fin_start}'")

  if end:
    if str(end).startswith("="):
      end_clause = f"ts <= '{str(end)[1:]}'"
    else:
      end_clause = f"ts < '{end}'"
    ts_where.append(end_clause)
    fin_ts_where.append(end_clause)
    shares_ts_where.append(end_clause)

  if fin_only_year:
    fin_ts_where.append("ts.month=12")

  ts_cond = f"and {' and '.join(ts_where)}" if ts_where else ""
  fin_ts_cond = f"and {' and '.join(fin_ts_where)}" if fin_ts_where else ""
  shares_ts_cond = f"and {' and '.join(shares_ts_where)}" if shares_ts_where else ""
  limit_cond = f" {limit}" if limit else ""
  fin_limit_cond = f" {fin_limit}" if fin_limit else ""
  shares_limit_cond = fin_limit_cond

  where_ = objs_where(objs)

  # Check financial fields requested
  fin_requested = fin_fields is not None and (
    isinstance(fin_fields, str) or len(fin_fields) > 0
  )
  if fin_requested:
    if fin_fields == "*" or fin_fields == ["*"]:
      logger.warning("fin_fields=['*'] requested: querying 180+ financial columns may consume significant memory.")
      fin_cols = "*"
    else:
      fin_cols = db_fields(fin_fields)
    if query_start:
      logger.info(
        f"financial query lookback: {fin_lookback} days (fin_start={fin_start}, fin_only_year={fin_only_year})"
      )

  sql = f"""
SELECT * FROM stock_kline_1d WHERE obj in ({where_}) {ts_cond}{limit_cond};
SELECT * FROM stock_dividend WHERE obj in ({where_}) {ts_cond}{limit_cond};
SELECT * FROM stock_shares WHERE obj in ({where_}) {shares_ts_cond}{shares_limit_cond};
"""
  logger.debug(f"start load data for {len(objs)} objs")
  t1 = time.time()
  try:
    table, objs, bars = client.query_join(
      sql,
      "stock_kline_1d",
      {"stock_dividend": "zero", "*": "backward"},
      objs[0],
    )
  except Exception as e:
    _check_dataset_min_ts(client, objs[0], target_start)
    raise RuntimeError(
      f"Failed to query market data from MSD ({msd_host}) for {len(objs)} obj(s) "
      f"in range [{target_start}, {end}]: {e}"
    ) from e

  if bars == 0 or table.rows == 0:
    _check_dataset_min_ts(client, objs[0], target_start)
    raise ValueError(
      f"No market data found in MSD ({msd_host}) for {len(objs)} obj(s) "
      f"(baseline='{objs[0]}') in range [{target_start}, {end}]."
    )

  data = dict((k, v) for k, v in table.to_list())
  logger.info(f"raw data build from msd, {len(objs)} objs, {bars} bars, {time.time() - t1:.3f}s")

  if data is None or "ts" not in data or len(data["ts"]) == 0:
    _check_dataset_min_ts(client, objs[0], target_start)
    raise ValueError(
      f"Empty 'ts' column returned from MSD ({msd_host}) for range [{target_start}, {end}]."
    )

  # Query and align stock_financial with statutory disclosure date fallback
  financial_frame = None
  if fin_requested:
    fin_sql = f"SELECT {fin_cols} FROM stock_financial WHERE obj in ({where_}) {fin_ts_cond}{fin_limit_cond};"
    try:
      res = client.query_concat(fin_sql)
      if "stock_financial" in res:
        table_fin, ranges = res["stock_financial"]
        d_fin = table_fin.to_dict()
        if len(d_fin.get("ts", [])) > 0:
          rep_ts_raw = d_fin["ts"]
          ann_ts_raw = to_statutory_disclosure_date(pl.Series("report_ts", rep_ts_raw)).to_numpy()
          
          # Build structured quarterly financial frame
          fin_objs_arr = np.repeat([r[0] for r in ranges], [r[2] - r[1] for r in ranges])
          ff_dict = {"obj": fin_objs_arr, "report_ts": rep_ts_raw, "ts": ann_ts_raw}
          fin_metric_cols = [c for c in d_fin.keys() if c != "ts"]
          for col in fin_metric_cols:
            ff_dict[col] = d_fin[col]
          financial_frame = pl.DataFrame(ff_dict)

          # Initialize metric columns in data
          for col in fin_metric_cols:
            data[col] = np.full(len(objs) * bars, np.nan, dtype=np.float64)
          data["report_ts"] = np.full(len(objs) * bars, np.datetime64("NaT", "us"), dtype="datetime64[us]")

          # Align per object using statutory disclosure date
          range_map = {r[0]: (r[1], r[2]) for r in ranges}
          for g, obj in enumerate(objs):
            kline_slice = slice(g * bars, (g + 1) * bars)
            if obj not in range_map:
              continue
            st, end_idx = range_map[obj]
            if st >= end_idx:
              continue
            obj_ann_ts = ann_ts_raw[st:end_idx]
            if len(obj_ann_ts) == 0:
              continue
            obj_rep_ts = rep_ts_raw[st:end_idx]
            kline_ts = data["ts"][kline_slice]

            idx = np.searchsorted(obj_ann_ts, kline_ts, side="right") - 1
            valid = idx >= 0
            safe_idx = np.maximum(idx, 0)

            for col in fin_metric_cols:
              val = d_fin[col][st:end_idx]
              if len(val) == 0:
                continue
              data[col][kline_slice] = np.where(valid, val[safe_idx], np.nan)
            data["report_ts"][kline_slice] = np.where(valid, obj_rep_ts[safe_idx], np.datetime64("NaT", "us"))
    except Exception as e:
      logger.error(f"failed to load stock_financial from msd: {e}", exc_info=True)
      raise

  for col in ["dividend", "transfer_shares", "right_shares", "right_price"]:
    if col not in data:
      data[col] = np.zeros_like(data["close"])

  dp = DataProvider(data, objs=objs, next_hook=next_hook_al)
  dp.fin_only_year = fin_only_year
  dp.financial_frame = financial_frame
  dp.warmup_bars = 0
  if warmup_bars > 0 and target_start and not isinstance(target_start, int) and dp.bars > 0:
    first_dates = dp.dates[:dp.bars]
    try:
      idx = find_date_index(first_dates, target_start)
      dp.warmup_bars = max(0, min(idx, dp.bars))
    except Exception as e:
      logger.warning(f"failed to compute exact warmup_bars from dates: {e}")
      dp.warmup_bars = max(0, min(int(warmup_bars), dp.bars))

  if dp.bars > 0 and len(dp.dates) > 0:
    start_dt_str = str(dp.dates[0])[:10]
    end_dt_str = str(dp.dates[dp.bars - 1])[:10]
    logger.info(
      f"MSD 数据载入完成: 可用历史区间为 {start_dt_str} 至 {end_dt_str} (共 {dp.bars} 根日K线, 预热条数: {dp.warmup_bars})"
    )
    if target_start and not isinstance(target_start, int):
      req_start_str = str(target_start)[:10]
      if req_start_str < start_dt_str:
        _check_dataset_min_ts(client, objs[0], target_start)
        try:
          gap_days = (
            datetime.date.fromisoformat(start_dt_str)
            - datetime.date.fromisoformat(req_start_str)
          ).days
        except ValueError:
          gap_days = 0
        if gap_days > 15:
          raise ValueError(
            f"Requested start '{req_start_str}' is earlier than dataset min_ts '{start_dt_str}' "
            f"(available range: {start_dt_str} to {end_dt_str})."
          )

  data["limit_ratio"] = get_limit_ratio(objs, repeat=dp.bars)

  # keep original unadjusted prices for realistic execution & limit calculation
  data["raw_open"] = data["open"].copy()
  data["raw_high"] = data["high"].copy()
  data["raw_low"] = data["low"].copy()
  data["raw_close"] = data["close"].copy()
  data["price"] = data["close"].copy()

  with al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN):
    # Calculate limit up/down on unadjusted prices with corporate-action reference price
    prev_raw_close = al.REF(data["price"], 1)
    denom = 1.0 + data["transfer_shares"] + data["right_shares"]
    denom = np.where(denom > 0, denom, 1.0)
    ex_ref_price = (
      prev_raw_close - data["dividend"] + data["right_shares"] * data["right_price"]
    ) / denom
    limit_up_price = np.round(ex_ref_price * (1.0 + data["limit_ratio"]), 2)
    limit_down_price = np.round(ex_ref_price * (1.0 - data["limit_ratio"]), 2)
    is_limit_up = (data["price"] == data["high"]) & (data["price"] >= limit_up_price)
    is_limit_down = (data["price"] == data["low"]) & (data["price"] <= limit_down_price)
    data["limited"] = np.where(is_limit_up, 1, np.where(is_limit_down, -1, 0))

    data["bw_price"] = al.BW_SPLIT(data["close"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
    data["fw_price"] = al.FW_SPLIT(data["close"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])

    logger.info("adjustment factors generated")
    # apply forward factor to all price related columns used to do technical analysis
    data["open"] = al.FW_SPLIT(data["open"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
    data["high"] = al.FW_SPLIT(data["high"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
    data["low"] = al.FW_SPLIT(data["low"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
    data["close"] = data["fw_price"]
    with np.errstate(divide="ignore", invalid="ignore"):
      raw_vwap = np.where(data["volume"] > 0, data["amount"] / data["volume"], np.nan)
      fw_factor = np.where(data["price"] > 0, data["fw_price"] / data["price"], 1.0)
    data["raw_vwap"] = raw_vwap
    data["vwap"] = raw_vwap * fw_factor
    logger.info("adjustment factors applied")

  if dp.bars == 0:
    raise ValueError("No data found for the given objs and date range.")

  return dp, objs
