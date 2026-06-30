import pymsd
from mbt import DataProvider
import alpha as al
import numpy as np
import logging

logger = logging.getLogger("msd_loader")



def next_hook_al(i: int, groups: int, bars: int):
  al.set_ctx(end=i + 1)


def load_data(
  msd_host: str, symbols: list[str], start: str, end: str
) -> tuple[DataProvider, list[str]]:
  """
  create a DataProvider for backtesting from msd database.

  Args:
    msd_host: msd server host
    symbols: symbols to backtest
    start: start date
    end: end date

  Returns:
    DataProvider and symbols
  """

  client = pymsd.create_msd_pandas(msd_host)

  dfs = client.load(
    objs=symbols,
    tables=["stock_kline_1d", "stock_dividend", "stock_financial"],
    start=start,
    end=end,
    join={"stock_dividend": "zero", "*": "backward"},
  )
  logger.info(f"data loaded from msd, {len(dfs)} symbols")

  data, symbols = client.adaptor.concat(dfs, base=symbols[0], join="nan")
  logger.info("data concatenated")
  if data is None or "ts" not in data:
    return DataProvider({"ts": np.array([])}, symbols=[]), []

  data = client.adaptor.to_numpy(data)

  dp = DataProvider(data, symbols=symbols, next_hook=next_hook_al)

  # keep original as price
  data["price"] = data["close"].copy()

  al.set_ctx(groups=dp.groups, flags=al.FLAG_SKIP_NAN)
  data["bw_price"] = al.BW_SPLIT(data["close"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
  data["fw_price"] = al.FW_SPLIT(data["close"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])

  logger.info("adjustment factors generated")
  # apply forward factor to all price related columns used to do technical analysis
  data["open"] = al.FW_SPLIT(data["open"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
  data["high"] = al.FW_SPLIT(data["high"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
  data["low"] = al.FW_SPLIT(data["low"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
  data["close"] = data["fw_price"]
  data["vwap"] = data["amount"] / data["volume"]
  # check if it is limited: 
  is_limit_up = (data["close"] == data["high"]) & (data["open"] == data["high"]) & (data["close"] > al.REF(data["close"], 1)) 
  is_limit_down = (data["close"] == data["low"]) & (data["open"] == data["low"]) & (data["close"] < al.REF(data["close"], 1))
  data["limited"] = np.where(is_limit_up, 1, np.where(is_limit_down, -1, 0))
  logger.info("adjustment factors applied")

  if dp.bars == 0:
    raise ValueError("No data found for the given symbols and date range.")

  # reset al context
  al.set_ctx(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=1)
  return dp, symbols
