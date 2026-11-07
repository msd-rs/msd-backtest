import time
import logging
import numpy as np
import alpha as al
import pymsd
from mbt import DataProvider
from mbt.shared import get_limit_ratio

logger = logging.getLogger("msd_loader")


def objs_where(objs: list[str]) -> str:
  return ",".join(map(lambda o: f'"{o}"', objs))


def next_hook_al(i: int, groups: int, bars: int):
  al.set_ctx(end=i + 1)


def load_data(
  msd_host: str, objs: list[str], start: str, end: str
) -> tuple[DataProvider, list[str]]:
  """
  create a DataProvider for backtesting from msd database.

  Args:
    msd_host: msd server host
    objs: objs to backtest
    start: start date
    end: end date

  Returns:
    DataProvider and objs
  """
  if not objs:
    return DataProvider({"ts": np.array([])}, objs=[]), []

  client = pymsd.MsdConnection(msd_host)

  ts_where = []
  limit = ""
  if isinstance(start, int):
    limit = f"limit -{start}"
  elif start:
    ts_where.append(f"ts >= '{start}'")

  if end:
    if str(end).startswith("="):
      ts_where.append(f"ts <= '{str(end)[1:]}'")
    else:
      ts_where.append(f"ts < '{end}'")

  ts_cond = f"and {' and '.join(ts_where)}" if ts_where else ""
  limit_cond = f" {limit}" if limit else ""

  where_ = objs_where(objs)
  sql = f"""
SELECT * FROM stock_kline_1d WHERE obj in ({where_}) {ts_cond}{limit_cond};
SELECT * FROM stock_dividend WHERE obj in ({where_}) {ts_cond}{limit_cond};
SELECT * FROM stock_shares WHERE obj in ({where_}) {ts_cond}{limit_cond};
SELECT * FROM stock_financial WHERE obj in ({where_}) {ts_cond}{limit_cond};
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
    logger.warning(f"failed to query_join from msd: {e}")
    return DataProvider({"ts": np.array([])}, objs=[]), []

  if bars == 0 or table.rows == 0:
    return DataProvider({"ts": np.array([])}, objs=[]), []

  data = dict((k, v) for k, v in table.to_list())
  logger.info(f"raw data build from msd, {len(objs)} objs, {bars} bars, {time.time() - t1:.3f}s")

  if data is None or "ts" not in data or len(data["ts"]) == 0:
    return DataProvider({"ts": np.array([])}, objs=[]), []

  for col in ["dividend", "transfer_shares", "right_shares", "right_price"]:
    if col not in data:
      data[col] = np.zeros_like(data["close"])

  dp = DataProvider(data, objs=objs, next_hook=next_hook_al)

  data["limit_ratio"] = get_limit_ratio(objs, repeat=dp.bars)

  # keep original unadjusted prices for realistic execution & limit calculation
  data["raw_open"] = data["open"].copy()
  data["price"] = data["close"].copy()

  al.set_ctx(groups=dp.groups, flags=al.FLAG_SKIP_NAN)

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

  # reset al context
  al.set_ctx(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=1)
  return dp, objs
