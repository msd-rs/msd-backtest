import pymsd
from mbt import DataProvider
import alpha as al
import numpy as np
import logging

logger = logging.getLogger("msd_loader")


def generate_adjustment_factors(df: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """
  计算前复权和后复权因子

  Arguments:
    df:  必须包含: ts(按时间升序), close(原始股价), dividend(每股分红), transfer_shares(每股送转股), right_shares(每股配股数), right_price(配股价)
  Returns:
    包含前复权因子 fw_factor 和后复权因子 bw_factor 的 DataFrame
  """

  # 2. 获取前一天的收盘价 (Pre-close)
  pre_close = al.REF(df["close"], 1)

  # 3. 计算除权除息日的变动比例 k
  # k = (前收盘 - 每股分红 + 配股比例 * 配股价) / (1 + 每股送股 + 每股转增 + 配股比例) / 前收盘
  # 我们只在有公司行为（分红、送转、配股）的日子计算 k，其余日子 k = 1.0

  # 分子：调整后的总价
  numerator = (
    pre_close - df["dividend"]+ df["right_shares"] * df["right_price"]
  )
  # 分母：调整后的总股本比例
  denominator = (
    1 + df["transfer_shares"] + df["transfer_shares"] + df["right_shares"] 
  ) * pre_close

  # 计算每日变动系数 k
  k = numerator / denominator

  # 处理没有公司行为的日期（k 设为 1.0）
  # 或者 pre_close 为空的第一行
  k = np.nan_to_num(k, nan=1.0)
  # 极端情况处理：如果没有发生除权行为，k 应该是 1.0
  k[
    (df["dividend"] == 0)
    & (df["transfer_shares"] == 0)
    & (df["right_shares"] == 0)
  ] = 1.0

  # 4. 生成后复权因子 (Backward Factor)
  # 逻辑：从上市首日开始，每次除权，因子都乘上 (1/k)
  # 后复权价 = 原始价 * 后复权因子
  bw_factor = np.cumprod(1.0 / k)

  # 5. 生成前复权因子 (Forward Factor)
  # 逻辑：以最后一天为 1.0，反向倒推。
  # 当前复权价 = 原始价 * 前复权因子
  # 前复权因子 = 后面所有 k 的累乘
  # 我们将 k 序列倒序后取 cumprod，再倒序回来
  fw_factor = np.cumprod(1.0 / k[::-1])[::-1]
  # fw_factor = np.roll(fw_factor, 1)

  return fw_factor, bw_factor


def generate_adjustment_factors_all(
  df: dict[str, np.ndarray], groups: int, bars: int
) -> dict[str, np.ndarray]:
  fw_factors = np.zeros(groups * bars)
  bw_factors = np.zeros(groups * bars)
  for i in range(groups):
    s = i * bars
    e = s + bars
    data = {
      "close": df["close"][s:e],
      "dividend": df["dividend"][s:e],
      "transfer_shares": df["transfer_shares"][s:e],
      "right_shares": df["right_shares"][s:e],
      "right_price": df["right_price"][s:e],
    }
    fw, bw = generate_adjustment_factors(data)
    fw_factors[s:e] = fw
    bw_factors[s:e] = bw
  df["fw_factor"] = fw_factors
  df["bw_factor"] = bw_factors
  return df


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
    tables=["stock_kline_1d", "stock_dividend"],
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

  data["bw_price"] = al.BW_SPLIT(data["close"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
  data["fw_price"] = al.FW_SPLIT(data["close"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])

  logger.info("adjustment factors generated")
  # apply forward factor to all price related columns used to do technical analysis
  data["open"] = al.FW_SPLIT(data["open"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
  data["high"] = al.FW_SPLIT(data["high"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
  data["low"] = al.FW_SPLIT(data["low"], data["dividend"], data["transfer_shares"], data["right_shares"], data["right_price"])
  data["close"] = data["fw_price"]
  logger.info("adjustment factors applied")

  if dp.bars == 0:
    raise ValueError("No data found for the given symbols and date range.")

  # reset al context
  al.set_ctx(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=1)
  return dp, symbols
