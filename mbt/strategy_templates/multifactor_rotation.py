# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""官方标准模版 ②：多因子评分全市场定期轮动模版 (Cross-Sectional Multi-Factor Periodic Rotation Template).

【适用场景】：
适用于宽基成分股（沪深300、中证500、中证1000）或全市场多因子量化选股，
融合动量（Momentum）、估值（Value / BP）与质量（Quality / ROE）三大经典因子，
按固定周期（如月度/每20交易日）进行全市场截面打分与等权/市值加权调仓轮动。

【架构契约】：
- 继承 `PortfolioStrategy`:
  - 共享资金池管理，`portfolio = True`，支持组合调仓引擎.
  - `exec_mode = "next_open"`：收盘打分确定权重，次日开盘集合竞价/开盘价执行换仓，完全规避当日收盘偷价未来函数.
  - 明确声明 `fin_fields = ["f003", "f205"]`，确保底层按需加载财务特征，避免内存浪费.
  - 声明 `warmup_bars = 60`，确保滚动因子平稳就绪.
- 核心最佳实践：
  * 复权隔离：动量因子基于前复权价格 `ctx.close` 计算；估值账面市值比 (BP) 严格使用未复权价格 `ctx.raw_close` 计算.
  * 稳健截面处理：使用 `np.errstate(divide="ignore", invalid="ignore")` 防御处理除零与缺失值；对单因子进行标准化 (Z-score).
  * 官方调仓 API：直接调用 `ctx.rebalance_objs(target_weights, tolerance=0.01)`，由引擎自动完成被剔除标的平仓与新选标的配资建仓.

【CLI 运行范例】：
# 1. 运行多标的定期轮动回测 (-p)，初始资金 100 万 (-c)
uv run python -m mbt.run.cli mbt/strategy_templates/multifactor_rotation.py -s SH600000 SH600519 SZ000001 SZ000858 -p -c 1000000 -b 2023-01-01 -e 2023-12-31 -S

# 2. 传入自定义参数 (-a) 调整因子权重与持仓数量
uv run python -m mbt.run.cli mbt/strategy_templates/multifactor_rotation.py -s SH600000 SH600519 SZ000001 SZ000858 -p -c 1000000 -b 2023-01-01 -e 2023-12-31 -a top_k=3 -a rebalance_interval=10 -S
"""

from __future__ import annotations

import logging
import numpy as np
import alpha as al

import mbt

logger = logging.getLogger(__name__)


def _calc_zscore(vals: np.ndarray) -> np.ndarray:
  """安全计算截面 Z-Score 标准化，防御 NaN 与极端零方差."""
  res = np.zeros_like(vals, dtype=np.float64)
  valid_mask = np.isfinite(vals)
  if not np.any(valid_mask):
    return res

  valid_vals = vals[valid_mask]
  mean = np.mean(valid_vals)
  std = np.std(valid_vals)
  if std > 1e-8:
    res[valid_mask] = (valid_vals - mean) / std
  return res


class MultiFactorRotationStrategy(mbt.PortfolioStrategy):
  """截面多因子定期轮动组合策略."""

  portfolio: bool = True
  exec_mode: str = "next_open"
  fin_fields: list[str] = ["f003", "f205"]
  warmup_bars: int = 60

  def __init__(
    self,
    top_k: int = 5,
    rebalance_interval: int = 20,
    weight_momentum: float = 0.4,
    weight_value: float = 0.3,
    weight_quality: float = 0.3,
  ):
    super().__init__()
    self.top_k = max(1, int(top_k))
    self.rebalance_interval = max(1, int(rebalance_interval))
    self.weight_momentum = float(weight_momentum)
    self.weight_value = float(weight_value)
    self.weight_quality = float(weight_quality)

    # 预计算动量指标缓存
    self.roc60: np.ndarray | None = None
    self.rebalance_count = 0

  def prepare(self, ctx: mbt.Context):
    """预计算时序动量特征 (60 日动量因子) 与初始化状态."""
    n_objs = ctx.n_objs
    bars = ctx.bars

    # 全量向量化预计算动量指标：(close - REF(close, 60)) / REF(close, 60)
    with al.Context(groups=n_objs, flags=al.FLAG_SKIP_NAN, start=0, end=bars):
      close = ctx.close
      ref_close = al.REF(close, 60)
      with np.errstate(divide="ignore", invalid="ignore"):
        self.roc60 = np.where(ref_close > 0.0, (close - ref_close) / ref_close, 0.0)

    self.rebalance_count = 0

  def execute(self, ctx: mbt.Context):
    """逐日驱动执行：在定期调仓日计算截面得分并执行权重重平衡."""
    # 调仓触发条件：超出预热期且达到调仓周期步长
    if ctx.is_warmup:
      return

    effective_bar = ctx.bar - ctx.warmup_bars
    if effective_bar % self.rebalance_interval != 0:
      return

    n_objs = ctx.n_objs
    current_bar = ctx.bar
    objs = ctx.objs()

    # 1. 提取当前截面各标的因子值
    # flat_idx = i * bars + current_bar
    base_indices = np.arange(n_objs) * ctx.bars + current_bar

    # 因子 1: 动量因子 (前复权价格变化率)
    mom_vals = self.roc60[base_indices]

    # 因子 2: 估值因子 BP (账面市值比 = 每股净资产 f003 / 真实未复权价格 raw_close)
    with np.errstate(divide="ignore", invalid="ignore"):
      raw_close = ctx.raw_close[base_indices]
      f003_val = ctx.f003[base_indices] if hasattr(ctx, "f003") else np.zeros(n_objs)
      val_bp = np.where((raw_close > 0.0) & (f003_val > 0.0), f003_val / raw_close, 0.0)

    # 因子 3: 质量因子 ROE (净资产收益率 f205)
    f205_val = ctx.f205[base_indices] if hasattr(ctx, "f205") else np.zeros(n_objs)
    qua_roe = np.where(np.isfinite(f205_val), f205_val, 0.0)

    # 2. 截面标准化 (Z-Score)
    z_mom = _calc_zscore(mom_vals)
    z_val = _calc_zscore(val_bp)
    z_qua = _calc_zscore(qua_roe)

    # 3. 综合加权评分
    total_scores = (
      self.weight_momentum * z_mom +
      self.weight_value * z_val +
      self.weight_quality * z_qua
    )

    # 4. 排序选取得分最高的 Top K 标的
    # 过滤无效标的（例如已停牌或无有效价格数据标的）
    valid_mask = np.isfinite(total_scores) & (ctx.close[base_indices] > 0.0)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
      return

    k = min(self.top_k, len(valid_indices))
    # 取得分最高的 top_k 个索引
    sorted_order = np.argsort(total_scores[valid_indices])[::-1]
    top_group_indices = valid_indices[sorted_order[:k]]

    # 5. 构建目标持仓权重字典并调用一等公民 rebalance_objs 引擎
    target_weight_each = 1.0 / k
    target_weights: dict[str, float] = {
      objs[g]: target_weight_each for g in top_group_indices
    }

    # 一行调用调仓：未在 target_weights 中的持仓自动清仓，新增或权重大于容差的标的自动配资买入
    ctx.rebalance_objs(target_weights, tolerance=0.01)
    self.rebalance_count += 1

  def finalize(self, ctx: mbt.Context) -> dict:
    """输出轮动调仓诊断报告."""
    return {
      "rebalance_count": self.rebalance_count,
      "top_k": self.top_k,
      "rebalance_interval": self.rebalance_interval,
    }
