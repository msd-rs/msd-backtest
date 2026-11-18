# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""双均线金叉死叉策略（Rust Fast-Path 版本）.

基于 `strategy/double_avg.py` 改写：
- 通过实现 `signals(ctx)` 显式信号钩子，`Runner` 在全量区间 `[0, bars)` 上一次性
  计算 `MA` / `CROSS` / `RCROSS`（复杂度由 O(N * T^2) 降至 O(N * T)），
  并直接调度 `alpha.sim.Simulator` 多线程 Rust 内核完成精确撮合与分红送股调整。
- 不触发逐 Bar Python `execute(ctx)` 回调，5000 股票 × 2000 日线规模可百毫秒级完成。

CLI 运行示例:
  uv run python -m mbt.run.cli strategy/double_avg_fast.py -s SH600000 SZ000001 -b 2023-01-01 -e 2023-12-31 -S
  uv run python -m mbt.run.cli strategy/double_avg_fast.py -s SH600000 -b 2023-01-01 -e 2023-12-31 -a fast_period=10 -a slow_period=30 -S
"""

import alpha as al
import numpy as np

import mbt

M_AUTHOR = "Gemini 4 Hydrogen"


class DoubleAvgFastStrategy(mbt.Strategy):
  """Fast-path dual moving-average crossover strategy."""

  def __init__(self, fast_period: int = 5, slow_period: int = 20):
    super().__init__()
    self.fast_period = int(fast_period)
    self.slow_period = int(slow_period)

  def signals(self, ctx: mbt.Context) -> tuple[np.ndarray, np.ndarray]:
    """Compute vectorized entry/exit signals once for the Rust Simulator."""
    close_price = ctx.data("close")

    fast_ma = al.MA(close_price, self.fast_period)
    slow_ma = al.MA(close_price, self.slow_period)

    buy_signals = al.CROSS(fast_ma, slow_ma)
    sell_signals = al.RCROSS(fast_ma, slow_ma)
    return buy_signals, sell_signals
