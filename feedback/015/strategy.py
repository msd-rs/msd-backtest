"""德瑞曼式「业绩为王 + 低市盈率」选股与半年度调仓策略（优化版）。

投资逻辑
--------
大卫·德瑞曼的逆向投资核心是：市场会系统性低估「业绩大幅改善、但估值仍然便宜」
的公司。原始版只要求「净利润高增长 + 低 PE」，在 A 股容易买到盈利见顶的周期股
与价值陷阱（持仓周期胜率仅 32.8%）。优化版在保留该核心的同时，增加两道确认：

1. 业绩为王：最新已披露报告期归母净利润同比 ≥ 20%，上年同期为正利润
   （排除亏损反转/低基数），扣非净利润为正，且增速 ≤ 300%（剔除异常值）。
2. 低估：PE = 未复权价 × 总股本 / TTM 归母净利润，要求 0 < PE ≤ 30。
3. 质量与趋势确认：年化 ROE ≥ 8%、资产负债率 ≤ 70%、经营现金流为正、
   营收增速 ≥ 0、120 日中期动量 ≥ 0。质量排除「便宜但没有现金流」的公司，
   趋势确认避免在下跌趋势中接飞刀。

因子与筛选
----------
* TTM 归母净利润 = 本期累计 + 上年全年 − 上年同期累计；同比增速自算，
  与 TTM 口径一致，不使用厂商增长率。
* 截面打分（百分位排名，去极值）：
  0.30 × rank(E/P) + 0.30 × rank(净利润增速) + 0.20 × rank(ROE) + 0.20 × rank(动量)。
* 可投资性：总市值 ≥ 30 亿、60 日中位成交额 ≥ 2000 万、当日未复权价有效。
* 基础池按未复权市值取全市场前 2000 只（只用可投资性、不用 alpha 条件），
  回测中在该宽池内动态重选，避免把起始日选股结果锁死为全周期池。

调仓机制
--------
* 只在 5/5、9/5 后的首个交易日调仓（上年年报 + 一季报、半年报）；
  去掉实证上负贡献的 11 月（三季报）窗口。
* 财报由 mbt 按法定披露日（4/30、8/31、10/31）做 as-of 对齐，成交用
  next_open（收盘产生信号、次日开盘成交），无未来函数。
* 20 只等权、1% 权重容差带；评分只决定「新买入谁」，已持仓只要仍通过
  同样的严格筛选就继续持有，不因排名小幅波动被换出。
* 不设止损、日频择时、行业轮动。use_market=1 可开启调仓日级大盘仓位控制
  （实测降回撤约 15pct，但牺牲约 60pct 收益，故默认关闭）。

财务字段（msd stock_financial，已核对语义）
  f097 归母净利润 / f088 扣非净利润 / f194 营收增长率(%) / f197 净利润增长率(%)
  f205 年化 ROE(%) / f223 资产负债率(%) / f108 经营现金流净额 /
  f234 经营现金净流量与净利润比率(%)

验证与开关
----------
2016-01-01 ~ 2025-12-31：收益 +302.63%（原始版 +52.91%），Sharpe 0.7383，
最大回撤 -38.31%，持仓周期胜率 51.38%。多起点、样本外 holdout、双倍费率
检验及完整消融过程见 optimize/report.md。
所有新增机制保留开关（use_quality / use_momentum / score_mode / windows /
hold_relaxed / mom_buffer / use_market），可用 -a key=value 覆盖，便于复核。
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import mbt

# f097 归母净利润 / f088 扣非净利润 / f194 主营收入增长率(%) /
# f197 净利润增长率(%) / f205 年化ROE(%) / f223 资产负债率(%) /
# f108 经营活动现金流量净额(元) / f234 经营现金净流量与净利润比率(%)
FIN_FIELDS = ["f097", "f088", "f194", "f197", "f205", "f223", "f108", "f234"]

A_PREFIXES = ("SH60", "SZ00", "SZ30", "SH68")
ALL_REBALANCE_DATES = ((5, 5), (9, 5), (11, 5))


def _pct_rank(values: np.ndarray) -> np.ndarray:
  """横截面百分位排名，取值 [0, 1]，越大越好。"""
  v = np.asarray(values, dtype=np.float64)
  n = len(v)
  if n <= 1:
    return np.ones(n, dtype=np.float64)
  order = np.argsort(v, kind="stable")
  rank = np.empty(n, dtype=np.float64)
  rank[order] = np.arange(n, dtype=np.float64)
  return rank / (n - 1)


def _safe_rank(values: np.ndarray) -> np.ndarray:
  """百分位排名；NaN/Inf 用截面中位数填充，防止 NaN 被排到最高分。"""
  v = np.asarray(values, dtype=np.float64)
  finite = v[np.isfinite(v)]
  fill = float(np.median(finite)) if len(finite) else 0.0
  return _pct_rank(np.nan_to_num(v, nan=fill, posinf=fill, neginf=fill))


def _season(date_str: str) -> tuple[int, int] | None:
  """把交易日映射到最近的财报披露窗口编号：1=5/5, 2=9/5, 3=11/5。"""
  if not date_str or len(date_str) < 10:
    return None
  year, month, day = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
  md = (month, day)
  if md >= (11, 5):
    return (year, 3)
  if md >= (9, 5):
    return (year, 2)
  if md >= (5, 5):
    return (year, 1)
  return (year - 1, 3)


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
  """长度为 len(x) 的简单移动均线，前 n-1 根为 NaN。"""
  v = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
  c = np.concatenate(([0.0], np.cumsum(v)))
  out = np.full(len(v), np.nan)
  if len(v) >= n:
    out[n - 1:] = (c[n:] - c[:-n]) / n
  return out


def _ttm_fundamentals(fin: pl.DataFrame) -> pl.DataFrame:
  """由季度累计财报还原 TTM 归母净利润与真实同比增速，并保留最新一期质量指标。"""
  base = (
    fin.with_columns([
      pl.col("report_ts").dt.year().alias("yr"),
      pl.col("report_ts").dt.month().alias("mo"),
    ])
    .sort(["obj", "yr", "mo"])
    .unique(subset=["obj", "yr", "mo"], keep="last")
    .sort(["obj", "yr", "mo"])
    .select(["obj", "yr", "mo"] + FIN_FIELDS)
    .rename({
      "f097": "np", "f088": "core", "f194": "rev_growth",
      "f197": "np_growth", "f205": "roe", "f223": "debt",
      "f108": "ocf", "f234": "ocf_ratio",
    })
  )

  latest = (
    base.group_by("obj", maintain_order=True)
    .tail(1)
    .rename({"np": "np_cur", "core": "core_cur", "yr": "cur_yr", "mo": "cur_mo"})
    .with_columns((pl.col("cur_yr") - 1).alias("prev_yr"))
  )

  prev_fy = (
    base.filter(pl.col("mo") == 12)
    .select(["obj", "yr", "np"])
    .rename({"yr": "prev_yr", "np": "np_fy"})
  )
  prev_same = base.select(["obj", "yr", "mo", "np"]).rename({
    "yr": "prev_yr", "mo": "cur_mo", "np": "np_same"
  })

  return (
    latest.join(prev_fy, on=["obj", "prev_yr"], how="left")
    .join(prev_same, on=["obj", "prev_yr", "cur_mo"], how="left")
    .with_columns([
      pl.when(pl.col("np_same") > 0)
      .then((pl.col("np_cur") - pl.col("np_same")) / pl.col("np_same") * 100.0)
      .otherwise(None)
      .alias("growth"),
      (pl.col("np_cur") + pl.col("np_fy") - pl.col("np_same")).alias("np_ttm"),
    ])
  )


class Selector(mbt.Selector):
  """宽口径基础池 + 质量/趋势硬筛选 + 截面「低 PE × 高增长」打分。"""

  fin_fields = FIN_FIELDS
  fin_only_year = False

  def __init__(
    self,
    pool_size: int = 2000,
    min_growth: float = 20.0,
    max_pe: float = 30.0,
    min_cap: float = 3e9,
    min_amount: float = 2e7,
    min_roe: float = 8.0,
    max_debt: float = 70.0,
    min_ocf_ratio: float = 0.0,
    min_rev_growth: float = 0.0,
    max_growth: float = 300.0,
    min_mom: float = 0.0,
    mom_window: int = 120,
    use_quality: int = 1,
    use_momentum: int = 1,
    score_mode: int = 1,
  ):
    super().__init__()
    self.score_mode = int(score_mode)
    self.pool_size = max(1, int(pool_size))
    self.min_growth = float(min_growth)
    self.max_pe = float(max_pe)
    self.min_cap = float(min_cap)
    self.min_amount = float(min_amount)
    self.min_roe = float(min_roe)
    self.max_debt = float(max_debt)
    self.min_ocf_ratio = float(min_ocf_ratio)
    self.min_rev_growth = float(min_rev_growth)
    self.max_growth = float(max_growth)
    self.min_mom = float(min_mom)
    self.mom_window = max(20, int(mom_window))
    self.use_quality = int(use_quality) != 0
    self.use_momentum = int(use_momentum) != 0
    self.factors = pl.DataFrame(schema={"obj": pl.String})

  def step00(self, stocks: list[str]) -> list[str]:
    """确定候选股票池：独立运行时默认全 A 股（剔除当前 ST）；回测中剔除指数等非个股。"""
    if not stocks:
      from mbt.select import A_STOCKS_EXCLUDE_ST
      return list(A_STOCKS_EXCLUDE_ST)
    return [s for s in stocks if s[:4] in A_PREFIXES]

  def step01(self, stocks: list[str]) -> list[str]:
    """仅独立选股时按未复权总市值取前 pool_size 只宽基础池，不叠加 alpha 条件。"""
    if self.is_backtest:
      return stocks
    if not stocks:
      return []
    snap = self.dp.load_snapshot(stocks)
    need = ("obj", "raw_close", "total_shares")
    if not all(c in snap.columns for c in need):
      return []
    snap = (
      snap.select(list(need))
      .filter((pl.col("raw_close") > 0) & (pl.col("total_shares") > 0))
      .with_columns((pl.col("raw_close") * pl.col("total_shares")).alias("mktcap"))
      .sort("mktcap", descending=True)
      .head(self.pool_size)
    )
    self.factors = snap
    return snap["obj"].to_list()

  def step02(self, stocks: list[str]) -> list[str]:
    """回测中取最新已披露财报，算 TTM 净利润/增速/PE，并叠加质量与规模硬筛选。"""
    if not self.is_backtest or not stocks:
      return stocks

    empty = pl.DataFrame(schema={"obj": pl.String})
    fin = self.dp.load_financial(stocks, FIN_FIELDS, only_year=False, lastN=9)
    if fin is None or len(fin) == 0:
      self.factors = empty
      return []

    snap = self.dp.load_snapshot(stocks)
    need = ("obj", "raw_close", "total_shares")
    if not all(c in snap.columns for c in need):
      self.factors = empty
      return []

    f = _ttm_fundamentals(fin)
    df = (
      f.join(snap.select(list(need)), on="obj", how="inner")
      .with_columns((pl.col("raw_close") * pl.col("total_shares")).alias("mktcap"))
      .with_columns(
        pl.when(pl.col("np_ttm") > 0)
        .then(pl.col("mktcap") / pl.col("np_ttm"))
        .otherwise(None)
        .alias("pe")
      )
    )
    conds = [
      pl.col("raw_close") > 0,
      pl.col("np_cur") > 0,
      pl.col("np_same") > 0,
      pl.col("core_cur") > 0,
      pl.col("np_fy").is_not_null(),
      pl.col("np_ttm") > 0,
      pl.col("growth") >= self.min_growth,
      pl.col("mktcap") >= self.min_cap,
      pl.col("pe") > 0,
      pl.col("pe") <= self.max_pe,
    ]
    if self.use_quality:
      conds += [
        pl.col("roe") >= self.min_roe,
        pl.col("debt") <= self.max_debt,
        pl.col("ocf") > 0,
        pl.col("ocf_ratio") >= self.min_ocf_ratio,
        pl.col("rev_growth") >= self.min_rev_growth,
        pl.col("growth") <= self.max_growth,
      ]
    df = df.filter(*conds)
    self.factors = df
    return df["obj"].to_list()

  def step03(self, stocks: list[str]) -> list[str]:
    """回测中用 60 日中位成交额与中期动量过滤，再按 50/50 的 E/P 与增速排名打分。"""
    if not self.is_backtest or not stocks:
      return stocks
    df = self.factors
    if df is None or "pe" not in df.columns or "growth" not in df.columns:
      self.factors = pl.DataFrame(schema={"obj": pl.String, "score": pl.Float64})
      return []

    n_bars = max(60, self.mom_window)
    panel = self.dp.load_kline(stocks, lastN=n_bars, baseline="SH000001")
    window = min(60, panel.bars)
    amount = panel.column("amount")[:, -window:]
    close = panel.column("close")
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      adv = np.nanmedian(amount, axis=1)
      lag = min(self.mom_window, panel.bars)
      momentum = close[:, -1] / close[:, -lag] - 1.0
    adv = np.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
    momentum = np.nan_to_num(momentum, nan=-1.0, posinf=0.0, neginf=-1.0)
    cs = panel.cross_section(adv=adv, mom=momentum)

    df = df.join(cs, on="obj", how="left").filter(pl.col("adv") >= self.min_amount)
    if self.use_momentum:
      df = df.filter(pl.col("mom") >= self.min_mom)
    if len(df) == 0:
      self.factors = pl.DataFrame(schema={"obj": pl.String, "score": pl.Float64})
      return []

    if self.score_mode == 1:
      score = (
        0.30 * _safe_rank(1.0 / df["pe"].to_numpy())
        + 0.30 * _safe_rank(df["growth"].to_numpy())
        + 0.20 * _safe_rank(df["roe"].to_numpy())
        + 0.20 * _safe_rank(df["mom"].to_numpy())
      )
    else:
      score = 0.5 * _safe_rank(1.0 / df["pe"].to_numpy()) \
        + 0.5 * _safe_rank(df["growth"].to_numpy())
    ranked = df.with_columns(pl.Series("score", score)).sort("score", descending=True)
    self.factors = ranked.select(
      ["obj", "score", "pe", "growth", "mktcap", "adv", "mom", "roe", "debt", "np_ttm"]
    )
    return self.factors["obj"].to_list()


class Strategy(mbt.PortfolioStrategy):
  """财报驱动的等权再平衡组合：严格买入、5/9 双窗口、可选增强与仓位控制。"""

  fin_fields = FIN_FIELDS
  fin_only_year = False
  fin_lookback_days = 1200
  warmup_bars = 60
  exec_mode = "next_open"

  def __init__(
    self,
    top_n: int = 20,
    pool_size: int = 2000,
    min_growth: float = 20.0,
    max_pe: float = 30.0,
    min_cap: float = 3e9,
    min_amount: float = 2e7,
    tolerance: float = 0.01,
    min_roe: float = 8.0,
    max_debt: float = 70.0,
    min_ocf_ratio: float = 0.0,
    min_rev_growth: float = 0.0,
    max_growth: float = 300.0,
    min_mom: float = 0.0,
    mom_window: int = 120,
    use_quality: int = 1,
    use_momentum: int = 1,
    score_mode: int = 1,
    mom_buffer: float = 0.0,
    hold_relaxed: int = 0,
    hold_min_growth: float = -10.0,
    hold_max_pe: float = 60.0,
    windows: str = "5,9",
    use_market: int = 0,
    market_mode: str = "index",
    bench_ma: int = 200,
    bear_exposure: float = 0.3,
  ):
    super().__init__()
    self.top_n = max(1, int(top_n))
    self.tolerance = max(0.0, float(tolerance))
    common = dict(
      pool_size=pool_size, min_growth=min_growth, max_pe=max_pe,
      min_cap=min_cap, min_amount=min_amount, min_roe=min_roe,
      max_debt=max_debt, min_ocf_ratio=min_ocf_ratio,
      min_rev_growth=min_rev_growth, max_growth=max_growth,
      min_mom=min_mom, mom_window=mom_window,
    )
    self.selector = Selector(
      **common, score_mode=score_mode,
      use_quality=use_quality, use_momentum=use_momentum
    )
    self.hold_relaxed = int(hold_relaxed) != 0
    self.mom_buffer = max(0.0, float(mom_buffer))
    self.hold_selector = None
    if self.hold_relaxed:
      hold = dict(common)
      hold.update(min_growth=hold_min_growth, max_pe=hold_max_pe)
      self.hold_selector = Selector(**hold, use_quality=0, use_momentum=0)
    elif self.mom_buffer > 0:
      loose = dict(common)
      loose.update(min_mom=min_mom - self.mom_buffer)
      self.hold_selector = Selector(
        **loose, score_mode=score_mode,
        use_quality=use_quality, use_momentum=use_momentum,
      )
    month2season = {5: 1, 9: 2, 11: 3}
    self.windows = tuple(month2season[int(x)] for x in str(windows).split(",") if x.strip())
    self.use_market = int(use_market) != 0
    self.market_mode = str(market_mode)
    self.bench_ma = max(20, int(bench_ma))
    self.bear_exposure = min(1.0, max(0.0, float(bear_exposure)))
    self._last_season: tuple[int, int] | None = None
    self._rebalance_count = 0
    self._holding_sum = 0
    self._expo_sum = 0.0
    self._last_candidates = 0
    self._risk_on: np.ndarray | None = None

  def prepare(self, ctx: mbt.Context):
    """预计算趋势状态；预热期结束前不交易。"""
    self._last_season = None
    self._rebalance_count = 0
    self._holding_sum = 0
    self._expo_sum = 0.0
    self._last_candidates = 0
    self._risk_on = None
    if not self.use_market:
      return
    panel = ctx.panel()
    close = panel.column("close")
    if self.market_mode == "pool":
      prev = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
      with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.where((prev > 0) & np.isfinite(prev) & np.isfinite(close),
                       close / prev - 1.0, np.nan)
      daily = np.nan_to_num(np.nanmean(ret, axis=0), nan=0.0)
      curve = np.cumprod(1.0 + daily)
    else:
      idx = next((i for i, o in enumerate(panel.objs)
                  if o.startswith("SH000") or o.startswith("SZ399")), None)
      if idx is None:
        return
      curve = np.asarray(close[idx], dtype=np.float64)
    ma = _rolling_mean(curve, self.bench_ma)
    with np.errstate(invalid="ignore"):
      self._risk_on = curve > ma

  def _exposure(self, bar: int) -> float:
    """调仓日的目标总仓位：趋势向上满仓，否则降到 bear_exposure。"""
    if not self.use_market or self._risk_on is None or bar >= len(self._risk_on):
      return 1.0
    return 1.0 if bool(self._risk_on[bar]) else self.bear_exposure

  def execute(self, ctx: mbt.Context):
    """按配置窗口重选并调仓；买入用严格条件，持有用宽口径复核。"""
    if ctx.is_warmup:
      return
    season = _season(ctx.date)
    if season is None or season == self._last_season:
      return
    first = self._last_season is None
    self._last_season = season
    if first:
      return
    if season[1] not in self.windows:
      return

    n_bars = max(60, self.selector.mom_window)
    ranked = list(ctx.run_selector(self.selector, lastN=n_bars))
    self._last_candidates = len(ranked)
    self._rebalance_count += 1
    expo = self._exposure(ctx.bar)
    self._expo_sum += expo

    if self.hold_selector is not None:
      keep_pool = set(ctx.run_selector(self.hold_selector, lastN=n_bars))
    else:
      keep_pool = set(ranked)

    if not ranked and not keep_pool:
      ctx.set_universe([])
      ctx.rebalance_objs({}, tolerance=self.tolerance)
      return

    held = set(ctx.account.held_objs)
    keep = [obj for obj in ranked if obj in held]
    if self.hold_selector is not None:
      keep += [obj for obj in held if obj in keep_pool and obj not in keep]
    keep = keep[: self.top_n]
    fresh = [obj for obj in ranked if obj not in keep][: self.top_n - len(keep)]
    chosen = keep + fresh

    if not chosen:
      ctx.set_universe([])
      ctx.rebalance_objs({}, tolerance=self.tolerance)
      return

    ctx.set_universe(chosen)
    weight = expo / len(chosen)
    ctx.rebalance_objs({obj: weight for obj in chosen}, tolerance=self.tolerance)
    self._holding_sum += len(chosen)

  def finalize(self, ctx: mbt.Context) -> dict | None:
    """输出调仓次数、平均持仓、平均仓位与开关状态，便于消融对照。"""
    return {
      "rebalance_count": self._rebalance_count,
      "avg_holdings": round(self._holding_sum / max(1, self._rebalance_count), 2),
      "avg_exposure": round(self._expo_sum / max(1, self._rebalance_count), 3),
      "last_candidate_pool": self._last_candidates,
      "rebalance_windows": [d for d in ALL_REBALANCE_DATES if self._season_id(d) in self.windows],
      "use_quality": self.selector.use_quality,
      "use_momentum": self.selector.use_momentum,
      "score_mode": self.selector.score_mode,
      "mom_buffer": self.mom_buffer,
      "hold_relaxed": self.hold_relaxed,
      "use_market": self.use_market,
    }

  @staticmethod
  def _season_id(day: tuple[int, int]) -> int:
    return {5: 1, 9: 2, 11: 3}[day[0]]
