import mbt
import polars as pl
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
    self.factors = pl.DataFrame()

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
      df = df.sort("ts")
      profits = df["f137"].tail(3).to_list()

      # 排除含有 None 或 NaN 的记录
      if any(p is None or np.isnan(p) for p in profits):
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
    self.factors = pl.DataFrame(profit_records)
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
        close = df["close"][-1]
        total_shares = df["total_shares"][-1]
        if close is not None and total_shares is not None:
          all_caps[sym] = close * total_shares

    if not all_caps:
      return []

    # 剔除 None/NaN 后的市值 Series
    cap_values = [v for v in all_caps.values() if v is not None and (not isinstance(v, float) or not np.isnan(v))]
    if not cap_values:
      return []
    cap_series = pl.Series("market_cap", cap_values)
    threshold = cap_series.quantile(1.0 - self.large_cap_percentile)

    # 过滤当前的股票集合
    selected_stocks = []
    for sym in stocks:
      if sym in all_caps:
        val = all_caps[sym]
        if val is not None and (not isinstance(val, float) or not np.isnan(val)) and val >= threshold:
          selected_stocks.append(sym)

    # 更新 factors DataFrame
    if not self.factors.is_empty():
      self.factors = self.factors.filter(pl.col("symbol").is_in(selected_stocks))
      self.factors = self.factors.with_columns(
        pl.col("symbol").map_elements(lambda x: all_caps.get(x), return_dtype=pl.Float64).alias("market_cap")
      )

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
    if not self.factors.is_empty():
      self.factors = self.factors.filter(pl.col("symbol").is_in(selected_stocks))
      self.factors = self.factors.with_columns([
        pl.col("symbol").map_elements(lambda x: last_closes.get(x), return_dtype=pl.Float64).alias("close"),
        pl.col("symbol").map_elements(lambda x: last_ma15s.get(x), return_dtype=pl.Float64).alias("ma15"),
      ])

    return selected_stocks
