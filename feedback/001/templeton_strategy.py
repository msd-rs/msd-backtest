"""约翰·邓普顿式「低价优质成长股」组合策略。

思路：
  1. Selector 先在全市场筛出「长期前景良好」的候选池：盈利为正、净资产为正、
     年化 ROE 较高、净利润与营收增长、负债可控、流动性充足、上市满一年。
  2. Strategy 每 20 个交易日重新给候选池打分，只在股价相对自身历史与全市场
     都明显低估、且市场情绪悲观时买入，最多持有 5 只，等权配置，其余时间持有不动。

财务字段口径（msd stock_financial）：
  f000 摊薄每股收益 / f003 每股净资产 / f194 主营收入增长率(%)
  f197 净利润增长率(%) / f205 年化净资产收益率(%) / f213 销售毛利率(%)
  f223 资产负债率(%) / f234 经营现金净流量与净利润的比率(%)
"""

import datetime as dt
import warnings

import alpha as al
import mbt
import numpy as np
import polars as pl
from mbt.select import A_STOCKS_EXCLUDE_ST

FIN_FIELDS = ["f000", "f003", "f194", "f197", "f205", "f213", "f223"]


class Selector(mbt.Selector):
  """构建「低估值、高增长、长期前景良好」的候选股票池。"""

  def __init__(
    self,
    min_roe: float = 8.0,
    min_profit_growth: float = 0.0,
    min_sales_growth: float = -5.0,
    max_debt: float = 95.0,
    min_price: float = 2.0,
    min_amount: float = 2e7,
    min_listed_days: int = 200,
    max_pool: int = 150,
    report_lag_days: int = 60,
  ):
    super().__init__()
    self.min_roe = min_roe
    self.min_profit_growth = min_profit_growth
    self.min_sales_growth = min_sales_growth
    self.max_debt = max_debt
    self.min_price = min_price
    self.min_amount = min_amount
    self.min_listed_days = min_listed_days
    self.max_pool = max_pool
    self.report_lag_days = report_lag_days
    self._snapshot: pl.DataFrame | None = None

  def step00(self, stocks: list[str]) -> list[str]:
    """以全部非 ST 的 A 股为初始集合；若指定了股票列表则沿用指定集合。"""
    if not stocks:
      stocks = list(A_STOCKS_EXCLUDE_ST)
    return list(dict.fromkeys(stocks))

  def step01(self, stocks: list[str]) -> list[str]:
    """按价格、成交额剔除仙股、低流动性股票和停牌股，取得最新行情快照。"""
    snap = self.dp.load_snapshot(stocks)
    snap = snap.with_columns(
      pl.col("close").fill_nan(None),
      pl.col("amount").fill_nan(None),
    )
    df = snap.filter(
      (pl.col("close") >= self.min_price)
      & (pl.col("amount") >= self.min_amount)
    )
    self._snapshot = df
    self.factors = df
    return df["obj"].to_list()

  def step02(self, stocks: list[str]) -> list[str]:
    """用已公告的最新财报筛选盈利、成长与财务稳健的公司，按估值/成长/质量综合分排序。"""
    fin = self.dp.load_financial(stocks, FIN_FIELDS, only_year=False, lastN=4)
    base = self.dp.date or dt.date.today().isoformat()
    cutoff = dt.datetime.strptime(base[:10], "%Y-%m-%d") - dt.timedelta(
      days=self.report_lag_days
    )
    fin = (
      fin.filter(pl.col("ts") <= cutoff)
      .group_by("obj", maintain_order=True)
      .last()
    )
    df = self._snapshot.join(fin, on="obj", how="inner").filter(
      (pl.col("f000") > 0)
      & (pl.col("f003") > 0)
      & (pl.col("f205") >= self.min_roe)
      & (pl.col("f197") >= self.min_profit_growth)
      & (pl.col("f194") >= self.min_sales_growth)
      & (pl.col("f223") <= self.max_debt)
      & (pl.col("f213") > 0)
    )
    df = df.with_columns(
      (pl.col("close") / pl.col("f003")).alias("pb"),
      (pl.col("close") / (pl.col("f003") * pl.col("f205") / 100)).alias("pe"),
    )
    # 估值/成长/盈利质量统一换算成横截面百分位再加权，避免低 PB 金融股独占候选池
    df = df.with_columns(
      (
        0.45 * pl.col("pe").rank(method="average", descending=True) / pl.col("obj").count()
        + 0.35 * pl.col("f197").rank(method="average") / pl.col("obj").count()
        + 0.20 * pl.col("f205").rank(method="average") / pl.col("obj").count()
      ).alias("quality_value_score"),
    )
    df = df.sort("quality_value_score", descending=True, nulls_last=True).head(
      max(self.max_pool * 2, self.max_pool)
    )
    self.factors = df
    return df["obj"].to_list()

  def step03(self, stocks: list[str]) -> list[str]:
    """用近一年日线剔除次新股与成交清淡的股票，最终保留最多 max_pool 只候选。"""
    panel = self.dp.load_kline(stocks, lastN=250, baseline="SH000001")
    kline = panel.column("close")
    listed_days = np.isfinite(kline).sum(axis=1)
    recent = panel.column("amount")[:, -60:]
    counts = np.isfinite(recent).sum(axis=1)
    avg_amount = np.nansum(recent, axis=1) / np.maximum(counts, 1)
    # 回测起点较早时可用历史 K 线可能不足一年，按比例放宽上市天数要求
    required_days = min(self.min_listed_days, max(int(panel.bars * 0.6), 1))
    tradable = (listed_days >= required_days) & (avg_amount >= self.min_amount)
    keep = [o for i, o in enumerate(panel.objs) if tradable[i]]
    df = self.factors.filter(pl.col("obj").is_in(keep)).head(self.max_pool)
    self.factors = df
    return df["obj"].to_list()


class Strategy(mbt.PortfolioStrategy):
  """每 period 个交易日调仓一次，买入被严重低估的优质成长股，最多持有 top_k 只。"""

  def __init__(
    self,
    period: int = 20,
    top_k: int = 5,
    warmup: int = 60,
    report_lag: int = 20,
    value_window: int = 120,
    min_roe: float = 5.0,
    min_profit_growth: float = 0.0,
    discount: float = 1.1,
    max_pe: float = 60.0,
    w_value: float = 0.40,
    w_growth: float = 0.25,
    w_quality: float = 0.20,
    w_pessimism: float = 0.15,
  ):
    super().__init__()
    self.period = max(1, int(period))
    self.top_k = max(1, int(top_k))
    self.warmup = max(1, int(warmup))
    self.report_lag = max(0, int(report_lag))
    self.value_window = max(20, int(value_window))
    self.min_roe = min_roe
    self.min_profit_growth = min_profit_growth
    self.discount = discount
    self.max_pe = max_pe
    self.w_value = w_value
    self.w_growth = w_growth
    self.w_quality = w_quality
    self.w_pessimism = w_pessimism
    self.n = 0
    self.score: np.ndarray | None = None
    self.valid: np.ndarray | None = None

  def prepare(self, ctx: mbt.Context):
    """一次性预计算全周期的估值、成长、质量与悲观程度因子。"""
    panel = ctx.panel()
    n, bars = panel.n_objs, panel.bars
    cols = set(panel.frame.columns)

    def get(name: str) -> np.ndarray:
      if name not in cols:
        return np.full(n * bars, np.nan, dtype=np.float64)
      return np.asarray(panel.buffer(name), dtype=np.float64)

    def lag(x: np.ndarray) -> np.ndarray:
      # 财报按报告期对齐，滞后约一个月再使用，规避公告延迟造成的前视偏差
      return al.REF(x, self.report_lag) if self.report_lag > 0 else x

    def rank(x: np.ndarray) -> np.ndarray:
      v = np.where(np.isfinite(x), x, np.nan)
      r = np.nan_to_num(al.CC_RANK(v), nan=0.5, posinf=0.5, neginf=0.5)
      return np.clip(r, 0.0, 1.0)

    price = get("price")
    close = get("close")
    limited = get("limited")
    eps = lag(get("f000"))
    bvps = lag(get("f003"))
    roe = lag(get("f205"))
    profit_g = lag(get("f197"))
    sales_g = lag(get("f194"))
    gross = lag(get("f213"))
    debt = lag(get("f223"))
    cash_quality = lag(get("f234"))

    with np.errstate(divide="ignore", invalid="ignore"):
      # 由年化 ROE 与每股净资产反推 TTM 每股收益，避开季报累计 EPS 的季节性
      eps_ttm = bvps * roe / 100.0
      pe = np.where((eps_ttm > 0) & (price > 0), price / eps_ttm, np.nan)
      pb = np.where((bvps > 0) & (price > 0), price / bvps, np.nan)
      pe = np.where((pe > 0) & (pe < 500), pe, np.nan)
      pb = np.where((pb > 0) & (pb < 50), pb, np.nan)
      peg = pe / np.maximum(profit_g, 5.0)
      pb_med = al.QUANTILE(pb, self.value_window, 0.5)
      pb_rel = pb / pb_med
      drawdown = close / al.HHV(close, self.value_window) - 1.0
      reversal = close / al.REF(close, 20) - 1.0

    value = (
      0.45 * (1 - rank(pe))
      + 0.35 * (1 - rank(pb))
      + 0.20 * (1 - rank(peg))
    )
    growth = 0.60 * rank(profit_g) + 0.40 * rank(sales_g)
    quality = (
      0.40 * rank(roe)
      + 0.20 * rank(gross)
      + 0.20 * (1 - rank(debt))
      + 0.20 * rank(cash_quality)
    )
    pessimism = (
      0.45 * (1 - rank(pb_rel))
      + 0.35 * (1 - rank(drawdown))
      + 0.20 * (1 - rank(reversal))
    )
    score = (
      self.w_value * value
      + self.w_growth * growth
      + self.w_quality * quality
      + self.w_pessimism * pessimism
    )

    pe2 = pe.reshape(n, bars)
    pb2 = pb.reshape(n, bars)
    pb_rel2 = pb_rel.reshape(n, bars)
    price2 = price.reshape(n, bars)
    limited2 = limited.reshape(n, bars)
    roe2 = roe.reshape(n, bars)
    profit_g2 = profit_g.reshape(n, bars)
    eps2 = eps.reshape(n, bars)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      market_pe = np.nanmedian(pe2, axis=0)

    # 严重低估：相对市场 PE 更低，且 PB 不高于自身历史中枢 × discount
    undervalued = (pe2 <= market_pe) & (pb_rel2 <= self.discount) & np.isfinite(pb_rel2)
    valid = (
      undervalued
      & np.isfinite(pe2)
      & (pe2 <= self.max_pe)
      & np.isfinite(pb2)
      & np.isfinite(price2)
      & (price2 > 0)
      & (limited2 == 0)
      & (roe2 >= self.min_roe)
      & (profit_g2 > self.min_profit_growth)
      & (eps2 > 0)
    )
    valid[0, :] = False  # objs[0] 为基准指数，不参与交易

    self.n = n
    self.score = score.reshape(n, bars)
    self.valid = valid

  def execute(self, ctx: mbt.Context):
    """每 20 个交易日按综合得分选出最多 5 只等权持有，其余交易日不调仓。"""
    # 框架未暴露公开的 bar 序号，调仓节奏依赖 runner 的当前步
    i = ctx.dp.i
    if i < self.warmup or (i - self.warmup) % self.period != 0:
      return
    assert self.score is not None and self.valid is not None

    weights = np.zeros(self.n, dtype=np.float64)
    candidates = np.flatnonzero(self.valid[:, i])
    if len(candidates) > 0:
      scores = np.where(np.isfinite(self.score[:, i]), self.score[:, i], -np.inf)
      chosen = candidates[np.argsort(-scores[candidates])[: self.top_k]]
      weights[chosen] = 1.0 / len(chosen)
    ctx.rebalance(weights)
