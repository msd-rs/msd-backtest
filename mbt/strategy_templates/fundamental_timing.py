"""官方标准模版 ①：基本面筛选 + 趋势择时模版 (Fundamental Screening + Trend Timing Template).

【适用场景】：
适用于价值成长投资、优质白马股、低波高股息及戴维斯双击（好公司 + 估值合理 + 右侧动量突破）。
单文件集成了选股器 (`Selector`) 与组合交易策略 (`PortfolioStrategy`)，支持单文件双层无缝联动。

【动态成分换血与白名单 (Universe Management)】：
本策略示范了在策略运行期间如何使用定时器或按季度调用 `ctx.run_selector()` 进行成分股动态换血：
1. 换血触发模式 (`rebalance_mode`):
   - `"quarterly"` (默认): 季度初/财报季动态换血，检测自然季度切换 (Q1~Q4)；
   - `"interval"`: 定时器周期换血，每隔 `rebalance_interval` 交易日 (如 60 日) 触发；
   - `"none"`: 保持初始静态池，回测期间不换血。
2. 无未来函数动态选股：
   - 策略调用 `picked = ctx.run_selector(self.selector, lastN=100)`，在回测切片数据上动态评估最新财报与估值。
   - 调用 `ctx.set_universe(picked)` 更新买入白名单，未入选标的被禁止新增建仓。
3. 淘汰标的处置机制 (`purge_outdated`):
   - `purge_outdated = True` (默认): 强制换血，若已有持仓跌出新白名单，立即平仓腾挪资金；
   - `purge_outdated = False`: 自然淘汰 (Runoff 模式)，允许旧持仓继续按均线和止损保护自然跑完离场。

【CLI 运行范例】：
# 1. 单文件双层初选组合回测 (使用初始选出标的池，期间按季度换血):
uv run python -m mbt.run.cli mbt/strategy_templates/fundamental_timing.py -b 2023-01-01 -e 2023-12-31 -p -c 1000000 -S

# 2. 全市场真·动态成分换血 (显式传入全市场 A_STOCKS 作为基础加载池，策略按季度全市场重选换血):
uv run python -m mbt.run.cli mbt/strategy_templates/fundamental_timing.py -s A_STOCKS -b 2023-01-01 -e 2023-12-31 -p -c 1000000 -a rebalance_mode=quarterly -S

# 3. 使用定时器每 40 个交易日换血一次 (-a rebalance_mode=interval -a rebalance_interval=40):
uv run python -m mbt.run.cli mbt/strategy_templates/fundamental_timing.py -b 2023-01-01 -e 2023-12-31 -p -c 1000000 -a rebalance_mode=interval -a rebalance_interval=40 -S
"""

from __future__ import annotations

import logging
import numpy as np
import polars as pl
import alpha as al

import mbt
from mbt.select import Panel

logger = logging.getLogger(__name__)


class FundamentalTimingSelector(mbt.Selector):
  """基本面优质成长与合理估值选股器 (单文件选股端)."""

  fin_fields: list[str] = ["f205", "f197", "f003"]

  def __init__(
    self,
    min_roe: float = 8.0,
    min_profit_growth: float = 10.0,
    max_pe: float = 35.0,
    **kwargs,
  ):
    super().__init__()
    self.min_roe = float(min_roe)
    self.min_profit_growth = float(min_profit_growth)
    self.max_pe = float(max_pe)
    self.factors = pl.DataFrame()

  def step00(self, stocks: list[str]) -> list[str]:
    """步骤 0：全市场股票池初始化与 ST 过滤."""
    if not self.is_backtest and not stocks:
      from mbt.select import A_STOCKS_EXCLUDE_ST
      stocks = A_STOCKS_EXCLUDE_ST
    return stocks

  def step01(self, stocks: list[str]) -> list[str]:
    """步骤 1：财务基本面质量初筛 (ROE 收益率与净利润同比增长率).

    通过底层 DataProvider 加载披露日法定对齐数据，杜绝未来函数.
    """
    if not stocks:
      return []

    # 加载最新报告期财务指标：f205 (净资产收益率 ROE), f197 (净利润同比增长率), f003 (每股净资产 BPS)
    fin_data = self.dp.load_financial(
      stocks,
      fields=["f205", "f197", "f003"],
      only_year=False,
      lastN=1,
    )

    if isinstance(fin_data, dict):
      rows = []
      for sym, df in fin_data.items():
        if df is not None and not df.is_empty():
          rows.append(df.with_columns(pl.lit(sym).alias("obj")))
      fin_data = pl.concat(rows) if rows else pl.DataFrame()

    if fin_data.is_empty():
      self.factors = pl.DataFrame()
      return []

    sym_col = "obj"
    fin_grouped = (
      fin_data.filter(pl.col(sym_col).is_in(stocks))
      .group_by(sym_col, maintain_order=True)
      .tail(1)
      .filter(
        (pl.col("f205") >= self.min_roe) &
        (pl.col("f197") >= self.min_profit_growth) &
        (pl.col("f003") > 0.0)
      )
    )

    selected = fin_grouped[sym_col].to_list()
    self.factors = fin_grouped.select([
      pl.col(sym_col).alias("obj"),
      pl.col("f205").alias("roe"),
      pl.col("f197").alias("profit_growth"),
      pl.col("f003").alias("bps"),
    ])
    return selected

  def step02(self, stocks: list[str]) -> list[str]:
    """步骤 2：合理估值筛选 (动态市盈率 PE <= max_pe).

    注意：估值指标必须使用真实未复权价格 (raw_close 或 snapshot 价格) 进行计算.
    """
    if not stocks:
      return []

    snapshot = self.dp.load_snapshot(stocks)
    if snapshot.is_empty():
      return stocks

    price_col = "raw_close" if "raw_close" in snapshot.columns else "close"
    merged = self.factors.join(
      snapshot.select(["obj", pl.col(price_col).alias("price")]),
      on="obj",
      how="inner",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
      # 简易年化 PE 估算: 价格 / 每股收益 EPS = 价格 / (BPS * ROE)
      merged = merged.with_columns(
        (pl.col("price") / (pl.col("bps") * (pl.col("roe") / 100.0))).alias("pe")
      )

    valid_pe = merged.filter((pl.col("pe") > 0.0) & (pl.col("pe") <= self.max_pe))
    self.factors = valid_pe
    return valid_pe["obj"].to_list()


class FundamentalTimingStrategy(mbt.PortfolioStrategy):
  """基本面与顺势右侧择时组合交易策略 (支持按季度/定时器全市场成分换血)."""

  portfolio: bool = True
  exec_mode: str = "close"
  warmup_bars: int = 60
  fin_fields: list[str] = ["f205", "f197", "f003"]

  def __init__(
    self,
    fast_period: int = 5,
    slow_period: int = 20,
    trend_period: int = 60,
    stop_loss_pct: float = 0.08,
    trailing_trigger: float = 0.12,
    trailing_stop_pct: float = 0.06,
    max_positions: int = 5,
    min_roe: float = 8.0,
    min_profit_growth: float = 10.0,
    max_pe: float = 35.0,
    rebalance_mode: str = "quarterly",
    rebalance_interval: int = 60,
    purge_outdated: bool = True,
    **kwargs,
  ):
    super().__init__()
    self.fast_period = int(fast_period)
    self.slow_period = int(slow_period)
    self.trend_period = int(trend_period)
    self.stop_loss_pct = float(stop_loss_pct)
    self.trailing_trigger = float(trailing_trigger)
    self.trailing_stop_pct = float(trailing_stop_pct)
    self.max_positions = max(1, int(max_positions))

    # 选股与换血参数
    self.min_roe = float(min_roe)
    self.min_profit_growth = float(min_profit_growth)
    self.max_pe = float(max_pe)
    self.rebalance_mode = str(rebalance_mode).lower()  # "quarterly", "interval", "none"
    self.rebalance_interval = max(1, int(rebalance_interval))
    self.purge_outdated = bool(purge_outdated)

    # 内置选股器实例（供 run_selector 动态换血复用）
    self.selector = FundamentalTimingSelector(
      min_roe=self.min_roe,
      min_profit_growth=self.min_profit_growth,
      max_pe=self.max_pe,
    )

    # 预计算指标缓存
    self.ma_fast: np.ndarray | None = None
    self.ma_slow: np.ndarray | None = None
    self.ma_trend: np.ndarray | None = None
    self.gold_cross: np.ndarray | None = None
    self.death_cross: np.ndarray | None = None

    # 运行期状态跟踪
    self.peak_rors: np.ndarray | None = None
    self.last_rebalance_quarter: int | None = None
    self.rebalance_count: int = 0
    self.exit_stats = {
      "stop_loss": 0,
      "trailing_stop": 0,
      "trend_break": 0,
      "universe_purge": 0,
    }

  def prepare(self, ctx: mbt.Context):
    """批量向量化预计算技术指标并预分配跟踪状态 (O(N * T) 复杂度)."""
    n_objs = ctx.n_objs
    bars = ctx.bars

    # 利用 alpha 向量化算子在全区间 [0, bars) 批量预计算
    with al.Context(groups=n_objs, flags=al.FLAG_SKIP_NAN, start=0, end=bars):
      close = ctx.close
      self.ma_fast = al.MA(close, self.fast_period)
      self.ma_slow = al.MA(close, self.slow_period)
      self.ma_trend = al.MA(close, self.trend_period)

      self.gold_cross = al.CROSS(self.ma_fast, self.ma_slow)
      self.death_cross = al.RCROSS(self.ma_fast, self.ma_slow)

    # 预分配各标的的最高收益率记录（用于移动止盈）
    self.peak_rors = np.full(n_objs, -np.inf, dtype=np.float64)
    self.last_rebalance_quarter = None
    self.rebalance_count = 0

  def execute(self, ctx: mbt.Context):
    """逐日驱动执行：检查换血调度、管理白名单、基于真实收益率出场与右侧顺势进场."""
    n_objs = ctx.n_objs
    current_bar = ctx.bar
    current_date_str = ctx.date
    target_pos_weight = 1.0 / self.max_positions

    # =========================================================================
    # 核心示范：使用定时器或按季度调用 ctx.run_selector() 进行成分换血
    # =========================================================================
    need_rebalance = False
    if not ctx.is_warmup and current_date_str:
      if self.rebalance_mode == "quarterly":
        # 自然季度模式：1-3月为Q1, 4-6月为Q2, 7-9月为Q3, 10-12月为Q4
        # 季初首个交易日触发成分股重选换血 (如 5月初年报/一季报出炉后、9月初半年报后、11月初三季报后)
        curr_month = int(current_date_str[5:7])
        curr_quarter = (curr_month - 1) // 3
        if self.last_rebalance_quarter is not None and curr_quarter != self.last_rebalance_quarter:
          need_rebalance = True
        self.last_rebalance_quarter = curr_quarter

      elif self.rebalance_mode == "interval":
        # 定时器模式：每隔固定的 rebalance_interval 个交易日触发一次
        effective_bars = current_bar - ctx.warmup_bars
        if effective_bars > 0 and effective_bars % self.rebalance_interval == 0:
          need_rebalance = True

    if need_rebalance:
      # 1. 在当前无未来函数的历史切片上重新执行 Selector 选股
      picked = ctx.run_selector(self.selector, lastN=100)

      # 2. 将选出的最新优质标的设置为当前组合买入白名单
      ctx.set_universe(picked)
      self.rebalance_count += 1
      logger.info(
        f"[{current_date_str}] 触发成分股换血: 候选池 {n_objs} 只中选出优质标的 {len(picked)} 只更新为 Universe 白名单"
      )

      # 3. 淘汰标的处置：若开启 purge_outdated，持仓中不在新白名单内的标的立即平仓腾挪资金
      if self.purge_outdated and ctx.universe is not None:
        for i in range(n_objs):
          sym = ctx.objs()[i]
          if ctx.position(i).shares > 0 and sym not in ctx.universe:
            ctx.sell(i, percent=1.0)
            self.exit_stats["universe_purge"] += 1
            self.peak_rors[i] = -np.inf

    # =========================================================================
    # 标的交易与风控流
    # =========================================================================
    # 统计当前实际持仓数量
    holding_count = sum(1 for i in range(n_objs) if ctx.position(i).shares > 0)

    for i in range(n_objs):
      sym = ctx.objs()[i]
      pos = ctx.position(i)
      has_pos = pos.shares > 0

      flat_idx = i * ctx.bars + current_bar
      close_price = ctx.close[flat_idx]
      trend_price = self.ma_trend[flat_idx]
      is_gold = bool(self.gold_cross[flat_idx])
      is_death = bool(self.death_cross[flat_idx])

      if has_pos:
        # 【核心规范】：必须通过 ctx.holding_ror(obj) 获取企业行动除权闭环后的真实收益率
        ror = ctx.holding_ror(i)
        self.peak_rors[i] = max(self.peak_rors[i], ror)

        # 规则 A：硬止损保护
        if ror <= -self.stop_loss_pct:
          ctx.sell(i, percent=1.0)
          self.exit_stats["stop_loss"] += 1
          self.peak_rors[i] = -np.inf
          holding_count -= 1
          continue

        # 规则 B：移动止盈锁定胜果 (盈利达到门槛后，从高点回撤触发)
        if self.peak_rors[i] >= self.trailing_trigger and (self.peak_rors[i] - ror) >= self.trailing_stop_pct:
          ctx.sell(i, percent=1.0)
          self.exit_stats["trailing_stop"] += 1
          self.peak_rors[i] = -np.inf
          holding_count -= 1
          continue

        # 规则 C：趋势走弱或均线死叉防守离场
        if is_death or (close_price < trend_price):
          ctx.sell(i, percent=1.0)
          self.exit_stats["trend_break"] += 1
          self.peak_rors[i] = -np.inf
          holding_count -= 1
          continue

      else:
        # 重置未持仓标的的浮盈记录
        self.peak_rors[i] = -np.inf

        # 规则 D：右侧顺势买入
        # 注意：白名单保护已在底层生效，此处也可显式验证标的是否在 universe 中
        in_universe = ctx.universe is None or sym in ctx.universe
        if in_universe and holding_count < self.max_positions:
          if is_gold and (close_price > trend_price):
            ctx.buy(i, percent=target_pos_weight)
            holding_count += 1

  def finalize(self, ctx: mbt.Context) -> dict:
    """输出语义诊断统计信息，沉淀于回测报告中."""
    return {
      "rebalance_count": self.rebalance_count,
      "rebalance_mode": self.rebalance_mode,
      "exit_stop_loss_count": self.exit_stats["stop_loss"],
      "exit_trailing_stop_count": self.exit_stats["trailing_stop"],
      "exit_trend_break_count": self.exit_stats["trend_break"],
      "exit_universe_purge_count": self.exit_stats["universe_purge"],
      "max_positions": self.max_positions,
    }
