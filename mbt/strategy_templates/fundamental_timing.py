"""官方标准模版 ①：基本面筛选 + 趋势择时模版 (Fundamental Screening + Trend Timing Template).

【适用场景】：
适用于价值成长投资、优质白马股、低波高股息及戴维斯双击（好公司 + 估值合理 + 右侧动量突破）。
单文件集成了选股器 (`Selector`) 与组合交易策略 (`PortfolioStrategy`)，支持单文件双层无缝联动。

【架构契约】：
- 选股端 (Selector):
  - 低频筛选：剔除 ST/停牌标的，基于法定披露日严格 backward as-of 对齐的财报数据初筛高 ROE 与高净利润增速标的.
  - 估值安全边际：结合未复权价格与财报真实指标过滤合理 PE/PB 估值区间.
- 交易端 (Strategy):
  - 右侧顺势介入：要求短期均线上穿中期均线（金叉）且价格运行在长期趋势生命线之上.
  - 一等公民出场与风控：
    * 硬止损：统一调用 `ctx.holding_ror(obj)` 判定跌幅止损，严禁手写混算除权成本.
    * 移动止盈 (Trailing Stop)：盈利启动后从高点回撤锁定胜果.
    * 趋势破位离场：跌破长期生命线即刻获利了结或防守避险.
  - 语义诊断：通过 `finalize()` 记录各退出原因的统计分布.

【CLI 运行范例】：
# 1. 仅运行单期选股并输出中间因子报表 (-f)
uv run python -m mbt.select.cli mbt/strategy_templates/fundamental_timing.py -d 2024-05-10 -f

# 2. 全联动组合回测 (-p)，初始资金 100 万 (-c)，输出 Markdown 精简诊断 (-S)
uv run python -m mbt.run.cli mbt/strategy_templates/fundamental_timing.py -b 2023-01-01 -e 2023-12-31 -p -c 1000000 -S

# 3. 覆盖自定义参数 (-a)
uv run python -m mbt.run.cli mbt/strategy_templates/fundamental_timing.py -b 2023-01-01 -e 2023-12-31 -p -c 1000000 -a min_roe=10.0 -a stop_loss_pct=0.06 -S
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
    if not stocks:
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
  """基本面与顺势右侧择时组合交易策略 (单文件策略端)."""

  portfolio: bool = True
  exec_mode: str = "close"
  warmup_bars: int = 60

  def __init__(
    self,
    fast_period: int = 5,
    slow_period: int = 20,
    trend_period: int = 60,
    stop_loss_pct: float = 0.08,
    trailing_trigger: float = 0.12,
    trailing_stop_pct: float = 0.06,
    max_positions: int = 5,
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

    # 预计算指标缓存
    self.ma_fast: np.ndarray | None = None
    self.ma_slow: np.ndarray | None = None
    self.ma_trend: np.ndarray | None = None
    self.gold_cross: np.ndarray | None = None
    self.death_cross: np.ndarray | None = None

    # 运行期状态跟踪
    self.peak_rors: np.ndarray | None = None
    self.exit_stats = {
      "stop_loss": 0,
      "trailing_stop": 0,
      "trend_break": 0,
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

  def execute(self, ctx: mbt.Context):
    """逐日驱动执行：基于持仓真实收益率出场与右侧顺势进场."""
    n_objs = ctx.n_objs
    current_bar = ctx.bar
    target_pos_weight = 1.0 / self.max_positions

    # 1. 统计当前持仓数量
    holding_count = sum(1 for i in range(n_objs) if ctx.position(i).shares > 0)

    # 2. 遍历各标的判断出场与入场信号
    for i in range(n_objs):
      pos = ctx.position(i)
      has_pos = pos.shares > 0

      # 扁平索引：标的 i 在当前 bar 的数据偏移
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

        # 规则 D：右侧顺势买入 (金叉且位于长期趋势生命线上方)
        if holding_count < self.max_positions:
          if is_gold and (close_price > trend_price):
            ctx.buy(i, percent=target_pos_weight)
            holding_count += 1

  def finalize(self, ctx: mbt.Context) -> dict:
    """输出语义诊断统计信息，沉淀于回测报告中."""
    return {
      "exit_stop_loss_count": self.exit_stats["stop_loss"],
      "exit_trailing_stop_count": self.exit_stats["trailing_stop"],
      "exit_trend_break_count": self.exit_stats["trend_break"],
      "max_positions": self.max_positions,
    }
