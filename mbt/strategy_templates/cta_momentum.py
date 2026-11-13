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
  * 价格语义隔离：形态突破与波幅计算使用前复权价格（`ctx.close`, `ctx.high`, `ctx.low`）；
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
    self.highest_price: np.ndarray | None = None
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
      self.buy_signals = (close > prev_channel_high) & (close > self.ma_trend)

      # 5. 通道跌破信号：今日收盘价跌破前一日的通道下轨
      prev_channel_low = al.REF(self.channel_low, 1)
      self.exit_signals = close < prev_channel_low

    # 预分配持仓最高价跟踪器
    self.highest_price = np.zeros(n_objs, dtype=np.float64)

  def execute(self, ctx: mbt.Context):
    """逐日驱动执行：极速 O(1) 读取信号并执行多层级动态出场与进场."""
    n_objs = ctx.n_objs
    current_bar = ctx.bar

    for i in range(n_objs):
      pos = ctx.position(i)
      has_pos = pos.shares > 0

      flat_idx = i * ctx.bars + current_bar
      cur_high = ctx.high[flat_idx]
      cur_close = ctx.close[flat_idx]
      cur_atr = self.atr[flat_idx]

      if has_pos:
        # 跟踪持仓期间的最高前复权价格（用于动态 ATR 追踪止损）
        self.highest_price[i] = max(self.highest_price[i], cur_high)

        # 规则 1：最大硬止损保护 (通过官方 holding_ror 严防送转股基准错配)
        ror = ctx.holding_ror(i)
        if ror <= -self.fixed_stop_loss:
          ctx.sell(i, percent=1.0)
          self.exit_stats["fixed_stop"] += 1
          self.highest_price[i] = 0.0
          continue

        # 规则 2：动态 ATR 波动率追踪止损 (从持仓最高价回撤 N 倍 ATR)
        trailing_stop_line = self.highest_price[i] - self.atr_stop_multiplier * cur_atr
        if cur_close < trailing_stop_line:
          ctx.sell(i, percent=1.0)
          self.exit_stats["atr_trailing"] += 1
          self.highest_price[i] = 0.0
          continue

        # 规则 3：通道下轨破位平仓
        if bool(self.exit_signals[flat_idx]):
          ctx.sell(i, percent=1.0)
          self.exit_stats["channel_exit"] += 1
          self.highest_price[i] = 0.0
          continue

      else:
        # 未持仓状态：重置最高价
        self.highest_price[i] = 0.0

        # 规则 4：通道突破进场
        if bool(self.buy_signals[flat_idx]):
          ctx.buy(i, percent=1.0)
          self.highest_price[i] = cur_high

  def finalize(self, ctx: mbt.Context) -> dict:
    """输出 CTA 出场统计与通道参数配置诊断."""
    return {
      "exit_fixed_stop_count": self.exit_stats["fixed_stop"],
      "exit_atr_trailing_count": self.exit_stats["atr_trailing"],
      "exit_channel_exit_count": self.exit_stats["channel_exit"],
      "channel_period": self.channel_period,
      "exit_channel_period": self.exit_channel_period,
    }
