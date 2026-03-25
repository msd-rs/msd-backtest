import pymsd
from mbt import DataProvider
import alpha as al
import numpy as np


def generate_adjustment_factors(df: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """
  计算前复权和后复权因子

  Arguments:
    df:  必须包含: ts(按时间升序), close, bonus, transfers, dividend, rightShare, rightPrice
  Returns:
    包含前复权因子 fw_factor 和后复权因子 bw_factor 的 DataFrame
  """

  # 2. 获取前一天的收盘价 (Pre-close)
  pre_close = df["close"]

  # 3. 计算除权除息日的变动比例 k
  # k = (前收盘 - 每股分红 + 配股比例 * 配股价) / (1 + 每股送股 + 每股转增 + 配股比例) / 前收盘
  # 我们只在有公司行为（分红、送转、配股）的日子计算 k，其余日子 k = 1.0

  # 分子：调整后的总价
  numerator = (
    pre_close - df["dividend"] / 10.0 + df["rightShare"] / 10.0 * df["rightPrice"]
  )
  # 分母：调整后的总股本比例
  denominator = (
    1 + df["bonus"] / 10.0 + df["transfers"] / 10.0 + df["rightShare"] / 10.0
  ) * pre_close

  # 计算每日变动系数 k
  k = numerator / denominator

  # 处理没有公司行为的日期（k 设为 1.0）
  # 或者 pre_close 为空的第一行
  k = np.nan_to_num(k, nan=1.0)
  # 极端情况处理：如果没有发生除权行为，k 应该是 1.0
  k[
    (df["dividend"] == 0)
    & (df["bonus"] == 0)
    & (df["transfers"] == 0)
    & (df["rightShare"] == 0)
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
      "bonus": df["bonus"][s:e],
      "transfers": df["transfers"][s:e],
      "rightShare": df["rightShare"][s:e],
      "rightPrice": df["rightPrice"][s:e],
    }
    fw, bw = generate_adjustment_factors(data)
    fw_factors[s:e] = fw
    bw_factors[s:e] = bw
  df["fw_factor"] = fw_factors
  df["bw_factor"] = bw_factors
  return df


def load_data(
  msd_host: str, symbols: list[str], start: str, end: str
) -> tuple[DataProvider, list[str]]:

  client = pymsd.create_msd_pandas(msd_host)

  dfs = client.load(
    objs=symbols,
    tables=["stock_kline_1d", "stock_dividend", "stock_shares"],
    start=start,
    end=end,
    join={"stock_dividend": "zero", "*": "backward"},
  )

  data, symbols = client.concat(dfs, base=symbols[0], join="nan")
  dp = DataProvider(data, symbols=symbols)
  al.set_ctx(groups=dp.groups, flags=al.FLAG_SKIP_NAN)

  data["price"] = data["close"].copy()  # keep original as price
  generate_adjustment_factors_all(data, dp.groups, dp.bars)
  data["open"] = data["open"] * data["bw_factor"]
  data["close"] = data["close"] * data["bw_factor"]
  data["high"] = data["high"] * data["bw_factor"]
  data["low"] = data["low"] * data["bw_factor"]

  return dp, symbols
