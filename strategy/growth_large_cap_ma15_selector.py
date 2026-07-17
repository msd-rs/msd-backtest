import mbt
import pandas as pd
import numpy as np
import alpha as al
from mbt.select import A_STOCKS_EXCLUDE_ST

class GrowthLargeCapMA15Selector(mbt.Selector):
  """
  选股器逻辑：
  1. 近三年盈利持续增长且为正 (使用 f137: 净利润)
  2. 大盘股 (市值排名前 20% 的股票)
  3. 最近 3 日内跌破 15 日均线
  """

  def __init__(self, large_cap_percentile: float = 0.2, break_days: int = 3):
    super().__init__()
    self.large_cap_percentile = large_cap_percentile
    self.break_days = break_days
    self.factors = pd.DataFrame()

  def step00(self, stocks: list[str]) -> list[str]:
    """
    第一步：筛选近三年盈利持续增长且为正的股票。
    使用 f137 (Net Profit) 作为净利润指标，每年 12-31 的年度数据进行对比。
    """
    if not stocks:
      stocks = A_STOCKS_EXCLUDE_ST

    # 加载近 3 年的年度财务数据
    fin_data = self.dp.load_financial(stocks, fields=["f137"], only_year=True, lastN=3)

    selected_stocks = []
    profit_records = []

    for sym in stocks:
      df = fin_data.get(sym)
      if df is None or len(df) < 3:
        continue

      # 按时间升序排序
      df = df.sort_values("ts")
      profits = df["f137"].values[-3:]

      # 排除含有 NaN 的记录
      if np.isnan(profits).any():
        continue

      # 盈利持续增长且为正 (P1 > 0 且 P2 > P1 且 P3 > P2)
      if profits[0] > 0 and profits[1] > profits[0] and profits[2] > profits[1]:
        selected_stocks.append(sym)
        profit_records.append({
          "symbol": sym,
          "net_profit_y1": profits[0],
          "net_profit_y2": profits[1],
          "net_profit_y3": profits[2],
        })

    # 初始化 factors 成员
    self.factors = pd.DataFrame(profit_records)
    return selected_stocks

  def step01(self, stocks: list[str]) -> list[str]:
    """
    第二步：筛选大盘股。
    计算所有正常交易的 A 股最新市值，取前 20% (由 large_cap_percentile 控制) 的股票。
    """
    if not stocks:
      return []

    # 获取全市场 A 股最新的收盘价和总股本，计算市值
    all_klines = self.dp.load_kline(A_STOCKS_EXCLUDE_ST, lastN=1)
    
    all_caps = {}
    for sym, df in all_klines.items():
      if len(df) > 0:
        row = df.iloc[-1]
        all_caps[sym] = row["close"] * row["total_shares"]

    if not all_caps:
      return []

    # 剔除 NaN 后的市值 Series
    cap_series = pd.Series(all_caps).dropna()
    threshold = cap_series.quantile(1.0 - self.large_cap_percentile)

    # 过滤当前的股票集合
    selected_stocks = []
    for sym in stocks:
      if sym in all_caps and not np.isnan(all_caps[sym]) and all_caps[sym] >= threshold:
        selected_stocks.append(sym)

    # 更新 factors DataFrame
    if not self.factors.empty:
      self.factors = self.factors[self.factors["symbol"].isin(selected_stocks)].copy()
      self.factors["market_cap"] = self.factors["symbol"].map(all_caps)

    return selected_stocks

  def step02(self, stocks: list[str]) -> list[str]:
    """
    第三步：筛选最近跌破 15 日均线的股票。
    加载最近 100 日 K 线，计算 15 日均线，判断最近 break_days 天内是否发生了跌破事件。
    """
    if not stocks:
      return []

    # 加载 K 线数据
    klines = self.dp.load_kline(stocks, lastN=100)

    selected_stocks = []
    last_closes = {}
    last_ma15s = {}

    # 设置 alpha 库单序列计算上下文
    al.set_ctx(groups=1, flags=al.FLAG_SKIP_NAN)

    for sym in stocks:
      df = klines.get(sym)
      if df is None or len(df) < 15:
        continue

      close = df["close"].to_numpy()
      ma15 = al.MA(close, 15)

      # 跌破信号：前一日收盘 >= 均线，今日收盘 < 均线
      death_cross = al.RCROSS(close, ma15)

      # 判断最近 break_days 天内是否有跌破信号
      lookback = min(self.break_days, len(death_cross))
      if death_cross[-lookback:].any():
        selected_stocks.append(sym)
        last_closes[sym] = close[-1]
        last_ma15s[sym] = ma15[-1]

    # 更新 factors DataFrame
    if not self.factors.empty:
      self.factors = self.factors[self.factors["symbol"].isin(selected_stocks)].copy()
      self.factors["close"] = self.factors["symbol"].map(last_closes)
      self.factors["ma15"] = self.factors["symbol"].map(last_ma15s)

    return selected_stocks
