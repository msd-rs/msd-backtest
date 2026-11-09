import mbt
import polars as pl
import numpy as np
import alpha as al
from mbt.select import A_STOCKS_EXCLUDE_ST, Panel

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

    # 加载近 3 年的年度财务数据（返回包含 ['obj', 'ts', 'f137'] 的长 DataFrame）
    fin_data = self.dp.load_financial(stocks, fields=["f137"], only_year=True, lastN=3)

    if isinstance(fin_data, dict):
      rows = []
      for sym, df in fin_data.items():
        if df is not None and not df.is_empty():
          rows.append(df.with_columns(pl.lit(sym).alias("obj")))
      fin_data = pl.concat(rows) if rows else pl.DataFrame()

    if fin_data.is_empty():
      self.factors = pl.DataFrame(schema={
        "obj": pl.String,
        "net_profit_y1": pl.Float64,
        "net_profit_y2": pl.Float64,
        "net_profit_y3": pl.Float64,
      })
      return []

    sym_col = "obj" if "obj" in fin_data.columns else "obj"
    fin_grouped = (
      fin_data.filter(pl.col(sym_col).is_in(stocks))
      .sort("ts")
      .group_by(sym_col, maintain_order=True)
      .agg([
        pl.col("f137").tail(3).alias("profits"),
        pl.len().alias("count"),
      ])
      .filter(pl.col("count") >= 3)
      .filter(
        pl.col("profits").list.get(0).is_not_null() & ~pl.col("profits").list.get(0).is_nan() &
        pl.col("profits").list.get(1).is_not_null() & ~pl.col("profits").list.get(1).is_nan() &
        pl.col("profits").list.get(2).is_not_null() & ~pl.col("profits").list.get(2).is_nan()
      )
      .filter(
        (pl.col("profits").list.get(0) > 0) &
        (pl.col("profits").list.get(1) > pl.col("profits").list.get(0)) &
        (pl.col("profits").list.get(2) > pl.col("profits").list.get(1))
      )
      .with_columns([
        pl.col(sym_col).alias("obj"),
        pl.col("profits").list.get(0).alias("net_profit_y1"),
        pl.col("profits").list.get(1).alias("net_profit_y2"),
        pl.col("profits").list.get(2).alias("net_profit_y3"),
      ])
      .select(["obj", "net_profit_y1", "net_profit_y2", "net_profit_y3"])
    )

    valid_map = {row["obj"]: row for row in fin_grouped.iter_rows(named=True)}
    selected_stocks = [s for s in stocks if s in valid_map]

    if selected_stocks:
      self.factors = pl.DataFrame([valid_map[s] for s in selected_stocks])
    else:
      self.factors = pl.DataFrame(schema={
        "obj": pl.String,
        "net_profit_y1": pl.Float64,
        "net_profit_y2": pl.Float64,
        "net_profit_y3": pl.Float64,
      })

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
    if isinstance(all_klines, Panel):
      close = all_klines.last("close")
      total_shares = all_klines.last("total_shares")
      for sym, c, s in zip(all_klines.objs, close, total_shares):
        if c is not None and s is not None and not np.isnan(c) and not np.isnan(s):
          all_caps[sym] = float(c * s)
    elif isinstance(all_klines, dict):
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
      self.factors = self.factors.filter(pl.col("obj").is_in(selected_stocks))
      self.factors = self.factors.with_columns(
        pl.col("obj").map_elements(lambda x: all_caps.get(x), return_dtype=pl.Float64).alias("market_cap")
      ).with_columns(
        (pl.col("market_cap") / pl.col("net_profit_y3")).alias("pe")
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

    if isinstance(klines, Panel):
      if klines.bars < 15:
        return []

      ma15_2d = klines.compute(al.MA, "close", 15)
      with klines.ctx():
        death_cross_2d = al.RCROSS(klines.buffer("close"), ma15_2d.ravel()).reshape(klines.n_objs, klines.bars)

      lookback = min(self.break_days, klines.bars)
      has_break = (death_cross_2d[:, -lookback:] > 0).any(axis=1)

      for i, sym in enumerate(klines.objs):
        if sym in stocks and has_break[i]:
          selected_stocks.append(sym)
          last_closes[sym] = float(klines.last("close")[i])
          last_ma15s[sym] = float(ma15_2d[i, -1])

    elif isinstance(klines, dict):
      # 设置 alpha 库单序列计算上下文，使用 al.Context 防止状态污染
      with al.Context(groups=1, flags=al.FLAG_SKIP_NAN):
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
      self.factors = self.factors.filter(pl.col("obj").is_in(selected_stocks))
      self.factors = self.factors.with_columns([
        pl.col("obj").map_elements(lambda x: last_closes.get(x), return_dtype=pl.Float64).alias("close"),
        pl.col("obj").map_elements(lambda x: last_ma15s.get(x), return_dtype=pl.Float64).alias("ma15"),
      ])

    return selected_stocks
