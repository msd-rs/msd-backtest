"""戴维斯型风格策略 (Davis Double Play Strategy).

基于谢尔比·戴维斯 (Shelby Davis)「戴维斯双击」投资哲学构建：
在公司估值较低且盈利增长健康稳健时介入，等待盈利加速与估值修复（双击）带来双重收益。
策略涵盖：
1. 基本面筛选 (DavisSelector)：低 PE/PB/PEG 安全边际 + 优质 ROE + 稳健营收/净利成长 + 正经营现金流与低负债排雷；
2. 组合交易 (DavisStrategy)：中长线定期性价比轮动再平衡 + 尾部硬止损 + 高位移动止盈 + 防戴维斯双杀保护。
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(override=True)

import logging
from typing import Any

import alpha as al
import mbt
import numpy as np
import polars as pl
from mbt.select import A_STOCKS_EXCLUDE_ST

logger = logging.getLogger(__name__)

M_AUTHOR = "Gemini 4 Hydrogen"


def get_board_sector(obj: str) -> str:
  """根据 A 股代码规则划分板块大类，用于分散度约束。"""
  if "688" in obj:
    return "STAR"
  if "300" in obj or "301" in obj:
    return "ChiNext"
  if obj.startswith("SH"):
    return "SH_Main"
  return "SZ_Main"


class DavisSelector(mbt.Selector):
  """戴维斯双击风格基本面选股器。"""

  fin_fields: list[str] = ["f205", "f197", "f194", "f003", "f002", "f213", "f223"]

  FACTOR_SCHEMA: dict[str, pl.DataType] = {
    "obj": pl.String,
    "price": pl.Float64,
    "pe": pl.Float64,
    "pb": pl.Float64,
    "peg": pl.Float64,
    "roe": pl.Float64,
    "profit_growth": pl.Float64,
    "growth_accel": pl.Float64,
    "rev_growth": pl.Float64,
    "gross_margin": pl.Float64,
    "ocfps": pl.Float64,
    "debt_ratio": pl.Float64,
    "score": pl.Float64,
  }

  def __init__(
    self,
    min_roe: float = 12.0,
    max_roe: float = 48.0,
    min_profit_growth: float = 10.0,
    max_profit_growth: float = 80.0,
    min_growth_accel: float = -12.0,
    min_rev_growth: float = 5.0,
    min_gross_margin: float = 20.0,
    min_pe: float = 4.8,
    max_pe: float = 20.0,
    max_pb: float = 4.0,
    max_peg: float = 1.2,
    max_debt_ratio: float = 60.0,
    roe_cap: float = 36.0,
    pe_floor: float = 5.0,
    accel_weight: float = 0.25,
    top_k: int = 30,
    max_per_sector: int = 10,
  ):
    super().__init__()
    self.min_roe = float(min_roe)
    self.max_roe = float(max_roe)
    self.min_profit_growth = float(min_profit_growth)
    self.max_profit_growth = float(max_profit_growth)
    self.min_growth_accel = float(min_growth_accel)
    self.min_rev_growth = float(min_rev_growth)
    self.min_gross_margin = float(min_gross_margin)
    self.min_pe = float(min_pe)
    self.max_pe = float(max_pe)
    self.max_pb = float(max_pb)
    self.max_peg = float(max_peg)
    self.max_debt_ratio = float(max_debt_ratio)
    self.roe_cap = float(roe_cap)
    self.pe_floor = float(pe_floor)
    self.accel_weight = float(accel_weight)
    self.top_k = max(1, int(top_k))
    self.max_per_sector = max(1, int(max_per_sector))
    self.factors: pl.DataFrame = pl.DataFrame(schema=self.FACTOR_SCHEMA)
    self._df: pl.DataFrame = pl.DataFrame()

  def step00(self, stocks: list[str]) -> list[str]:
    """步骤 0: 初始化候选池，加载最新两期财报与当前未复权快照，计算 PE/PB 与盈利加速度。"""
    if not self.is_backtest and not stocks:
      stocks = A_STOCKS_EXCLUDE_ST
    if not stocks:
      self._df = pl.DataFrame()
      return []

    fin_data = self.dp.load_financial(
      stocks,
      fields=self.fin_fields,
      only_year=False,
      lastN=2,
    )
    if isinstance(fin_data, dict):
      rows = [
        df.with_columns(pl.lit(sym).alias("obj"))
        for sym, df in fin_data.items()
        if df is not None and not df.is_empty()
      ]
      fin_data = pl.concat(rows) if rows else pl.DataFrame()

    if fin_data.is_empty():
      self._df = pl.DataFrame()
      return []

    fin_sub = fin_data.filter(pl.col("obj").is_in(stocks))
    if "report_ts" in fin_sub.columns:
      fin_sub = fin_sub.sort(["obj", "report_ts"])
    elif "ts" in fin_sub.columns:
      fin_sub = fin_sub.sort(["obj", "ts"])

    latest = fin_sub.group_by("obj", maintain_order=True).tail(1)
    prev = (
      fin_sub.group_by("obj", maintain_order=True)
      .head(1)
      .select(["obj", pl.col("f197").alias("prev_f197")])
    )

    snapshot = self.dp.load_snapshot(stocks)
    if snapshot.is_empty():
      self._df = pl.DataFrame()
      return []

    price_col = "raw_close" if "raw_close" in snapshot.columns else "close"
    merged = (
      latest.join(prev, on="obj", how="left")
      .join(snapshot.select(["obj", pl.col(price_col).alias("price")]), on="obj", how="inner")
      .filter(
        pl.col("price").is_not_null()
        & (pl.col("price") > 0.0)
        & pl.col("f003").is_not_null()
        & (pl.col("f003") > 0.0)
        & pl.col("f205").is_not_null()
        & (pl.col("f205") > 0.0)
      )
      .with_columns([
        (pl.col("price") / (pl.col("f003") * (pl.col("f205") / 100.0))).alias("pe"),
        (pl.col("price") / pl.col("f003")).alias("pb"),
        (pl.col("f197") - pl.col("prev_f197").fill_null(pl.col("f197"))).alias("growth_accel"),
      ])
    )

    self._df = merged
    return merged["obj"].to_list()

  def step01(self, stocks: list[str]) -> list[str]:
    """步骤 1: 低估值安全边际初筛 (剔除极低 PE 周期顶部假象，限定 PE 与 PB 合理区间)。"""
    if self._df.is_empty() or not stocks:
      self._df = pl.DataFrame()
      return []

    df = self._df.filter(
      pl.col("obj").is_in(stocks)
      & pl.col("pe").is_not_null()
      & (pl.col("pe") >= self.min_pe)
      & (pl.col("pe") <= self.max_pe)
      & pl.col("pb").is_not_null()
      & (pl.col("pb") > 0.0)
      & (pl.col("pb") <= self.max_pb)
    )
    self._df = df
    return df["obj"].to_list()

  def step02(self, stocks: list[str]) -> list[str]:
    """步骤 2: 盈利能力与成长性筛选 (ROE 区间防周期暴利陷阱、净利增速与盈利加速度、营收/毛利与 PEG)。"""
    if self._df.is_empty() or not stocks:
      self._df = pl.DataFrame()
      return []

    df = (
      self._df.filter(
        pl.col("obj").is_in(stocks)
        & (pl.col("f205") >= self.min_roe)
        & (pl.col("f205") <= self.max_roe)
        & pl.col("f197").is_not_null()
        & (pl.col("f197") >= self.min_profit_growth)
        & (pl.col("f197") <= self.max_profit_growth)
        & pl.col("growth_accel").is_not_null()
        & (pl.col("growth_accel") >= self.min_growth_accel)
        & pl.col("f194").is_not_null()
        & (pl.col("f194") >= self.min_rev_growth)
        & pl.col("f213").is_not_null()
        & (pl.col("f213") >= self.min_gross_margin)
      )
      .with_columns((pl.col("pe") / pl.col("f197")).alias("peg"))
      .filter(
        pl.col("peg").is_not_null()
        & (pl.col("peg") > 0.0)
        & (pl.col("peg") <= self.max_peg)
      )
    )
    self._df = df
    return df["obj"].to_list()

  def step03(self, stocks: list[str]) -> list[str]:
    """步骤 3: 财务安全与现金流排雷 (每股经营现金流为正且资产负债率可控)。"""
    if self._df.is_empty() or not stocks:
      self._df = pl.DataFrame()
      return []

    df = self._df.filter(
      pl.col("obj").is_in(stocks)
      & pl.col("f002").is_not_null()
      & (pl.col("f002") > 0.0)
      & pl.col("f223").is_not_null()
      & (pl.col("f223") <= self.max_debt_ratio)
    )
    self._df = df
    return df["obj"].to_list()

  def step04(self, stocks: list[str]) -> list[str]:
    """步骤 4: 戴维斯双击综合评分排序 (含 ROE 封顶、盈利加速度加权与 PE 下限保护) 与板块分散度筛选。"""
    if self._df.is_empty() or not stocks:
      self.factors = pl.DataFrame(schema=self.FACTOR_SCHEMA)
      return []

    df = (
      self._df.filter(pl.col("obj").is_in(stocks))
      .with_columns(
        (
          (
            pl.col("f205").clip(upper_bound=self.roe_cap)
            * (
              1.0
              + pl.col("f197") / 100.0
              + self.accel_weight * pl.col("growth_accel").clip(-20.0, 50.0) / 100.0
            )
          )
          / pl.col("pe").clip(lower_bound=self.pe_floor)
        ).alias("score")
      )
      .sort("score", descending=True)
    )

    selected_objs: list[str] = []
    sector_counts: dict[str, int] = {}
    for row in df.iter_rows(named=True):
      s_obj = row["obj"]
      sec = get_board_sector(s_obj)
      if sector_counts.get(sec, 0) < self.max_per_sector:
        selected_objs.append(s_obj)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(selected_objs) >= self.top_k:
          break

    if not selected_objs:
      self.factors = pl.DataFrame(schema=self.FACTOR_SCHEMA)
      return []

    rank_map = {obj: idx for idx, obj in enumerate(selected_objs)}
    self.factors = (
      df.filter(pl.col("obj").is_in(selected_objs))
      .select([
        pl.col("obj").cast(pl.String),
        pl.col("price").cast(pl.Float64),
        pl.col("pe").cast(pl.Float64),
        pl.col("pb").cast(pl.Float64),
        pl.col("peg").cast(pl.Float64),
        pl.col("f205").cast(pl.Float64).alias("roe"),
        pl.col("f197").cast(pl.Float64).alias("profit_growth"),
        pl.col("growth_accel").cast(pl.Float64),
        pl.col("f194").cast(pl.Float64).alias("rev_growth"),
        pl.col("f213").cast(pl.Float64).alias("gross_margin"),
        pl.col("f002").cast(pl.Float64).alias("ocfps"),
        pl.col("f223").cast(pl.Float64).alias("debt_ratio"),
        pl.col("score").cast(pl.Float64),
      ])
      .with_columns(pl.col("obj").replace_strict(rank_map).alias("_rank"))
      .sort("_rank")
      .drop("_rank")
    )
    return selected_objs


class DavisStrategy(mbt.PortfolioStrategy):
  """戴维斯型风格中长线价值成长组合策略。"""

  portfolio: bool = True
  exec_mode: str = "close"
  warmup_bars: int = 60
  fin_fields: list[str] = ["f205", "f197", "f194", "f003", "f002", "f213", "f223"]

  def __init__(
    self,
    rebalance_period: int = 40,
    max_positions: int = 6,
    max_hold_per_sector: int = 4,
    stop_loss_pct: float = 0.23,
    trailing_trigger: float = 0.65,
    trailing_stop_pct: float = 0.18,
    sell_max_pe: float = 35.0,
    trend_bonus: float = 1.08,
    exec_accel_weight: float = 0.15,
    min_roe: float = 12.0,
    max_roe: float = 48.0,
    min_profit_growth: float = 10.0,
    max_profit_growth: float = 80.0,
    min_growth_accel: float = -12.0,
    min_rev_growth: float = 5.0,
    min_gross_margin: float = 20.0,
    min_pe: float = 4.8,
    max_pe: float = 20.0,
    max_pb: float = 4.0,
    max_peg: float = 1.2,
    max_debt_ratio: float = 60.0,
    roe_cap: float = 36.0,
    pe_floor: float = 5.0,
    accel_weight: float = 0.25,
    top_k: int = 30,
    max_per_sector: int = 10,
  ):
    super().__init__()
    self.rebalance_period = max(1, int(rebalance_period))
    self.max_positions = max(1, int(max_positions))
    self.max_hold_per_sector = max(1, int(max_hold_per_sector))
    self.stop_loss_pct = float(stop_loss_pct)
    self.trailing_trigger = float(trailing_trigger)
    self.trailing_stop_pct = float(trailing_stop_pct)
    self.sell_max_pe = float(sell_max_pe)
    self.trend_bonus = float(trend_bonus)
    self.exec_accel_weight = float(exec_accel_weight)

    self.selector = DavisSelector(
      min_roe=min_roe,
      max_roe=max_roe,
      min_profit_growth=min_profit_growth,
      max_profit_growth=max_profit_growth,
      min_growth_accel=min_growth_accel,
      min_rev_growth=min_rev_growth,
      min_gross_margin=min_gross_margin,
      min_pe=min_pe,
      max_pe=max_pe,
      max_pb=max_pb,
      max_peg=max_peg,
      max_debt_ratio=max_debt_ratio,
      roe_cap=roe_cap,
      pe_floor=pe_floor,
      accel_weight=accel_weight,
      top_k=top_k,
      max_per_sector=max_per_sector,
    )

    self.ma20: np.ndarray | None = None
    self.ma60: np.ndarray | None = None
    self.peak_rors: np.ndarray | None = None
    self.rebalance_count: int = 0
    self.exit_reasons: dict[str, int] = {
      "stop_loss": 0,
      "trailing_stop": 0,
      "double_kill_guard": 0,
      "rebalance_rotation": 0,
    }

  def prepare(self, ctx: mbt.Context):
    """预先向量化计算均线指标与初始化运行时统计状态。"""
    n_objs = ctx.n_objs
    bars = ctx.bars

    with al.Context(groups=n_objs, flags=al.FLAG_SKIP_NAN, start=0, end=bars):
      close = ctx.close
      self.ma20 = al.MA(close, 20)
      self.ma60 = al.MA(close, 60)

    self.peak_rors = np.full(n_objs, -np.inf, dtype=np.float64)
    self.rebalance_count = 0
    self.exit_reasons = {
      "stop_loss": 0,
      "trailing_stop": 0,
      "double_kill_guard": 0,
      "rebalance_rotation": 0,
    }

  def execute(self, ctx: mbt.PortfolioContext):
    """每日执行持仓风控与定期戴维斯性价比轮动。"""
    if ctx.is_warmup:
      return

    bar = ctx.bar
    b_idx = getattr(ctx.account, "benchmark_idx", None)

    # 1. 每日持仓风控：硬止损 + 高位移动止盈 + 防戴维斯双杀保护
    for obj in list(ctx.account.held_objs):
      g = ctx.index_of(obj)
      idx = g * ctx.bars + bar
      c = ctx.close[idx]
      ma60_v = self.ma60[idx] if self.ma60 is not None else np.nan
      ror = ctx.holding_ror(g)
      self.peak_rors[g] = max(self.peak_rors[g], ror)

      if ror <= -self.stop_loss_pct:
        ctx.sell(g)
        self.exit_reasons["stop_loss"] += 1
        self.peak_rors[g] = -np.inf
        continue

      if (
        self.peak_rors[g] >= self.trailing_trigger
        and (self.peak_rors[g] - ror) >= self.trailing_stop_pct
      ):
        ctx.sell(g)
        self.exit_reasons["trailing_stop"] += 1
        self.peak_rors[g] = -np.inf
        continue

      g_prof = ctx.f197[idx] if hasattr(ctx, "f197") else np.nan
      if (
        not np.isnan(g_prof)
        and g_prof < -25.0
        and not np.isnan(ma60_v)
        and c < ma60_v * 0.95
      ):
        ctx.sell(g)
        self.exit_reasons["double_kill_guard"] += 1
        self.peak_rors[g] = -np.inf
        continue

    # 2. 定期戴维斯性价比再平衡轮动
    eff_bar = bar - ctx.warmup_bars
    if eff_bar % self.rebalance_period != 0:
      return

    fin_frame = getattr(ctx.dp, "financial_frame", None)
    if (
      ctx.n_objs > self.selector.top_k + 10
      and fin_frame is not None
      and len(fin_frame) > 0
    ):
      picked = ctx.run_selector(self.selector, lastN=100)
      if len(picked) >= self.max_positions:
        ctx.set_universe(picked)

    scored: list[tuple[float, str, int]] = []
    for g, obj in enumerate(ctx.objs()):
      if b_idx is not None and g == b_idx:
        continue
      if ctx.universe is not None and obj not in ctx.universe:
        continue

      idx = g * ctx.bars + bar
      c = ctx.close[idx]
      raw_c = ctx.raw_close[idx] if hasattr(ctx, "raw_close") else ctx.price[idx]
      bps = ctx.f003[idx] if hasattr(ctx, "f003") else np.nan
      roe = ctx.f205[idx] if hasattr(ctx, "f205") else np.nan
      g_prof = ctx.f197[idx] if hasattr(ctx, "f197") else np.nan
      g_rev = ctx.f194[idx] if hasattr(ctx, "f194") else np.nan
      ocf = ctx.f002[idx] if hasattr(ctx, "f002") else np.nan
      ma60_v = self.ma60[idx] if self.ma60 is not None else np.nan

      if (
        np.isnan(c)
        or c <= 0.0
        or np.isnan(raw_c)
        or raw_c <= 0.0
        or np.isnan(bps)
        or bps <= 0.0
        or np.isnan(roe)
        or roe <= 6.0
      ):
        continue
      if np.isnan(g_prof) or g_prof < 0.0:
        continue
      if not np.isnan(g_rev) and g_rev < -5.0:
        continue
      if np.isnan(ocf) or ocf <= 0.0:
        continue

      accel = 0.0
      if bar >= 65:
        prev_gp = ctx.f197[idx - 65]
        if not np.isnan(prev_gp):
          accel = g_prof - prev_gp

      pe = raw_c / (bps * (roe / 100.0))
      if pe <= 4.0 or pe > self.sell_max_pe:
        continue

      t_mult = self.trend_bonus if (not np.isnan(ma60_v) and c >= ma60_v) else 1.0
      score = (
        (
          min(roe, self.selector.roe_cap)
          * (
            1.0
            + np.clip(g_prof, 0.0, 100.0) / 100.0
            + self.exec_accel_weight * np.clip(accel, -20.0, 50.0) / 100.0
          )
        )
        / max(self.selector.pe_floor, pe)
      ) * t_mult
      scored.append((float(score), obj, g))

    scored.sort(key=lambda x: x[0], reverse=True)
    top: list[tuple[float, str, int]] = []
    sec_counts: dict[str, int] = {}
    for item in scored:
      sec = get_board_sector(item[1])
      if sec_counts.get(sec, 0) < self.max_hold_per_sector:
        top.append(item)
        sec_counts[sec] = sec_counts.get(sec, 0) + 1
        if len(top) >= self.max_positions:
          break
    if not top:
      return

    target_objs = {obj for _, obj, _ in top}
    for held in ctx.account.held_objs:
      if held not in target_objs:
        self.exit_reasons["rebalance_rotation"] += 1

    weight = 1.0 / len(top)
    target_weights = {obj: weight for _, obj, _ in top}
    ctx.rebalance_objs(target_weights, tolerance=0.02)
    self.rebalance_count += 1

    for _, obj, g in top:
      if not ctx.account.has_position(obj):
        self.peak_rors[g] = 0.0

  def finalize(self, ctx: mbt.Context) -> dict[str, Any]:
    """输出策略运行与退出原因诊断报告。"""
    return {
      "rebalance_count": self.rebalance_count,
      "max_positions": self.max_positions,
      "rebalance_period": self.rebalance_period,
      "exit_reasons": self.exit_reasons,
    }
