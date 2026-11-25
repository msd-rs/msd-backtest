# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""官方标准模版 ③：CTA 纯量价均线/通道跟踪模版 (CTA Price-Volume Momentum & Channel Breakout Template).

【适用场景】：
适用于经典趋势跟踪、唐奇安通道突破（海龟交易法则变体）与均线动量跟踪策略。
支持单标的独立回测以及多标的组合并发回测。

【架构契约】：
- 极致性能设计 (Zero-Overhead in on_bar):
  * 在 `prepare()` 阶段，利用 `alpha` 与 `numpy` 将通道极值 (HHV/LLV)、真实波幅 (ATR)、大势均线 (MA)
    及突破信号全部完成向量化预计算.
  * 在 `execute()` 阶段，单步执行仅做 O(1) 状态判断与信号读取，实现全市场秒级回测.
- 核心最佳实践与风控：
  * `exec_mode = "next_open"`：收盘突破生成信号，次日开盘价撮合成交，贴合实盘并彻底消除前视偏差.
  * 多层级风控协同：
    1. 动态 ATR 波动率追踪止损：以买入后的最高价 - N * ATR 建立动态跟踪防线，让利润奔跑的同时锁住回撤.
    2. 兜底硬止损：统一调用官方一等公民 API `ctx.holding_ror(obj)` 触发最大亏损截断.
    3. 通道下轨破位离场：跌破 N 日最低价顺势离场.
  * 价格语义隔离：形态突破与波幅计算使用等比后复权价格（`ctx.close`, `ctx.high`, `ctx.low`）；
    成本等效比对使用 `pos.adj_cost_price` 或 `ctx.cost_price(obj, adjusted=True)`.

【CLI 运行范例】：
# 1. 单标的独立回测 (-s)，输出精简 Markdown 报告 (-S)
uv run python -m mbt.run.cli mbt/strategy_templates/cta_momentum.py -s SH600000 -b 2023-01-01 -e 2023-12-31 -S

# 2. 多标的共享资金组合回测 (-p)，初始资金 100 万 (-c)
uv run python -m mbt.run.cli mbt/strategy_templates/cta_momentum.py -s SH600000 SZ000001 -p -c 1000000 -b 2023-01-01 -e 2023-12-31 -S

# 3. 传入自定义通道与 ATR 参数 (-a)
uv run python -m mbt.run.cli mbt/strategy_templates/cta_momentum.py -s SH600000 -b 2023-01-01 -e 2023-12-31 -a channel_period=30 -a atr_stop_multiplier=2.5 -S
"""

from __future__ import annotations

import logging
import numpy as np
import alpha as al

import mbt

logger = logging.getLogger(__name__)

M_AUTHOR = ""  # 编写策略时建议定义模块常量 M_AUTHOR，填入当前模型名称


class CtaMomentumStrategy(mbt.Strategy):
  """唐奇安通道突破与动态 ATR 追踪止损 CTA 策略."""

  exec_mode: str = "next_open"
  warmup_bars: int = 40

  def __init__(
    self,
    channel_period: int = 20,
    exit_channel_period: int = 10,
    atr_period: int = 14,
    atr_stop_multiplier: float = 2.0,
    fixed_stop_loss: float = 0.06,
  ):
    super().__init__()
    self.channel_period = max(2, int(channel_period))
    self.exit_channel_period = max(2, int(exit_channel_period))
    self.atr_period = max(2, int(atr_period))
    self.atr_stop_multiplier = float(atr_stop_multiplier)
    self.fixed_stop_loss = float(fixed_stop_loss)

    # 预计算向量化指标矩阵
    self.channel_high: np.ndarray | None = None
    self.channel_low: np.ndarray | None = None
    self.ma_trend: np.ndarray | None = None
    self.atr: np.ndarray | None = None
    self.buy_signals: np.ndarray | None = None
    self.exit_signals: np.ndarray | None = None

    # 运行期状态跟踪
    self.offsets: np.ndarray | None = None
    self.highest_price: np.ndarray | None = None
    self.pending_exit: np.ndarray | None = None
    self.pending_buy: np.ndarray | None = None
    self.exit_stats = {
      "fixed_stop": 0,
      "atr_trailing": 0,
      "channel_exit": 0,
    }

  def prepare(self, ctx: mbt.Context):
    """向量化预计算所有通道、均线与 ATR 波动率指标 (一次性全局加速)."""
    n_objs = ctx.n_objs
    bars = ctx.bars

    with al.Context(groups=n_objs, flags=al.FLAG_SKIP_NAN, start=0, end=bars):
      high = ctx.high
      low = ctx.low
      close = ctx.close

      # 1. 唐奇安通道：N 日最高价与 M 日最低价
      self.channel_high = al.HHV(high, self.channel_period)
      self.channel_low = al.LLV(low, self.exit_channel_period)

      # 2. 大势过滤均线 (如 2 * channel_period)
      self.ma_trend = al.MA(close, self.channel_period * 2)

      # 3. 真实波幅 (True Range) 与 ATR
      ref_close = al.REF(close, 1)
      tr1 = high - low
      tr2 = np.abs(high - ref_close)
      tr3 = np.abs(low - ref_close)
      tr = np.maximum(tr1, np.maximum(tr2, tr3))
      self.atr = al.MA(tr, self.atr_period)

      # 4. 突破信号：今日收盘价突破前一日的通道上轨，且站在大势生命线上方
      prev_channel_high = al.REF(self.channel_high, 1)
      self.buy_signals = np.asarray((close > prev_channel_high) & (close > self.ma_trend), dtype=bool)

      # 5. 通道跌破信号：今日收盘价跌破前一日的通道下轨
      prev_channel_low = al.REF(self.channel_low, 1)
      self.exit_signals = np.asarray(close < prev_channel_low, dtype=bool)

    # 预分配截面索引偏移与运行期状态数组
    self.offsets = np.arange(n_objs, dtype=np.int64) * bars
    self.highest_price = np.zeros(n_objs, dtype=np.float64)
    self.pending_exit = np.zeros(n_objs, dtype=bool)
    self.pending_buy = np.zeros(n_objs, dtype=bool)
    self.exit_stats = {
      "fixed_stop": 0,
      "atr_trailing": 0,
      "channel_exit": 0,
    }

  def execute(self, ctx: mbt.Context):
    """逐日驱动执行：截面向量化读取持仓状态，执行多层级动态出场与进场."""
    if ctx.is_warmup:
      return

    assert self.offsets is not None
    assert self.highest_price is not None
    assert self.pending_exit is not None
    assert self.pending_buy is not None
    assert self.atr is not None
    assert self.buy_signals is not None
    assert self.exit_signals is not None

    # 批量获取全截面持仓股数与除权闭环收益率，避免逐股循环调用 ctx.position(i)
    shares = ctx.holding_shares()
    rors = np.nan_to_num(ctx.holding_rors(), nan=0.0)
    held = shares > 0.0
    flat = ~held

    self.pending_exit &= held
    self.highest_price[flat] = 0.0
    pending_buy_prev = self.pending_buy.copy()
    self.pending_buy[:] = False

    idx = self.offsets + ctx.bar
    cur_high = ctx.high[idx]
    cur_close = ctx.close[idx]
    cur_atr = self.atr[idx]

    if np.any(held):
      # 跟踪持仓期间的最高复权价格（用于动态 ATR 追踪止损）
      self.highest_price[held] = np.maximum(self.highest_price[held], cur_high[held])

    manageable = held & ~self.pending_exit
    if np.any(manageable):
      # 规则 1：最大硬止损保护 (基于 holding_rors 严防送转股基准错配)
      fixed_stop = manageable & (rors <= -self.fixed_stop_loss)
      # 规则 2：动态 ATR 波动率追踪止损 (从持仓最高价回撤 N 倍 ATR)
      trailing_stop_line = self.highest_price - self.atr_stop_multiplier * cur_atr
      atr_trailing = (
        manageable
        & ~fixed_stop
        & np.isfinite(trailing_stop_line)
        & (cur_close < trailing_stop_line)
      )
      # 规则 3：通道下轨破位平仓
      channel_exit = manageable & ~fixed_stop & ~atr_trailing & self.exit_signals[idx]
      sell_mask = fixed_stop | atr_trailing | channel_exit

      if np.any(sell_mask):
        ctx.sell(sell_mask, percent=1.0)
        self.pending_exit[sell_mask] = True
        self.highest_price[sell_mask] = 0.0
        self.exit_stats["fixed_stop"] += int(np.count_nonzero(fixed_stop))
        self.exit_stats["atr_trailing"] += int(np.count_nonzero(atr_trailing))
        self.exit_stats["channel_exit"] += int(np.count_nonzero(channel_exit))

    # 规则 4：通道突破进场
    buy_mask = flat & ~pending_buy_prev & self.buy_signals[idx]
    if np.any(buy_mask):
      ctx.buy(buy_mask, percent=1.0)
      self.pending_buy[buy_mask] = True
      self.highest_price[buy_mask] = cur_high[buy_mask]

  def finalize(self, ctx: mbt.Context) -> dict:
    """输出 CTA 出场统计与通道参数配置诊断."""
    return {
      "exit_fixed_stop_count": self.exit_stats["fixed_stop"],
      "exit_atr_trailing_count": self.exit_stats["atr_trailing"],
      "exit_channel_exit_count": self.exit_stats["channel_exit"],
      "channel_period": self.channel_period,
      "exit_channel_period": self.exit_channel_period,
    }
