# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""基于持仓收益率的均线首仓 + 越跌越买 + 止盈轮动策略.

【策略逻辑】：
1. 首仓建仓：空仓状态下，当收盘价低于 60 日均线（`close < MA60`）时，买入当前可用资金的 5%。
2. 阶梯补仓：持仓状态下，每当持仓收益率（`ctx.holding_ror`）进一步降低 10%
   （即触及 -10%、-20%、-30% ...）时，再次买入当前可用资金的 5%，直至可用资金不足。
3. 止盈清仓：持仓状态下，当持仓收益率达到 +20% 时，全部卖出清仓，重置补仓阈值并等待下一次
   收盘价低于 60 日均线时重新建仓。

【架构与风控规范】：
- 持仓收益率统一调用 `ctx.holding_ror(g)`，底层自动兼容现金分红与送转股成本稀释，
  避免高送转除权导致虚假暴跌并误触发补仓。
- 技术均线 `MA60` 在 `prepare()` 阶段基于前复权 `ctx.close` 一次性向量化预计算。

【CLI 运行示例】：
  uv run python -m mbt.run.cli strategy/holding_ror_grid.py -s SH600000 -b 2023-01-01 -e 2023-12-31 -S
  uv run python -m mbt.run.cli strategy/holding_ror_grid.py -s SH600000 SZ000001 -b 2023-01-01 -e 2023-12-31 -a buy_percent=0.05 -a take_profit_pct=0.20 -S
"""

from __future__ import annotations

import alpha as al
import numpy as np

import mbt

M_AUTHOR = "Gemini 4 Hydrogen"


class HoldingRorGridStrategy(mbt.Strategy):
  """均线首仓 + 持仓收益每降 10% 买入 5% 可用资金 + 盈利 20% 全部止盈策略."""

  warmup_bars: int = 60

  def __init__(
    self,
    ma_period: int = 60,
    buy_percent: float = 0.05,
    add_drop_pct: float = 0.10,
    take_profit_pct: float = 0.20,
  ):
    super().__init__()
    self.ma_period = max(1, int(ma_period))
    self.buy_percent = float(buy_percent)
    self.add_drop_pct = float(add_drop_pct)
    self.take_profit_pct = float(take_profit_pct)

    self.ma: np.ndarray | None = None
    self.below_ma: np.ndarray | None = None
    self.next_add_ror: np.ndarray | None = None
    self.pending_exit: np.ndarray | None = None
    self.stats = {
      "initial_entries": 0,
      "add_on_buys": 0,
      "take_profit_exits": 0,
    }

  def prepare(self, ctx: mbt.Context):
    """Precompute MA indicator over full history with strict window length."""
    n_objs = ctx.n_objs
    bars = ctx.bars
    with al.Context(groups=n_objs, flags=al.FLAG_STRICTLY_CYCLE, start=0, end=bars):
      close = ctx.close
      self.ma = al.MA(close, self.ma_period)
      self.below_ma = ((~np.isnan(self.ma)) & (close < self.ma)).reshape(n_objs, bars)

    self.next_add_ror = np.full(n_objs, -self.add_drop_pct, dtype=np.float64)
    self.pending_exit = np.zeros(n_objs, dtype=bool)
    self.stats = {
      "initial_entries": 0,
      "add_on_buys": 0,
      "take_profit_exits": 0,
    }

  def execute(self, ctx: mbt.Context):
    """Vectorized per-bar check for take-profit, stepped add-on buys, and MA60 initial entry."""
    if ctx.is_warmup:
      return
    assert self.below_ma is not None
    assert self.next_add_ror is not None
    assert self.pending_exit is not None

    cur_bar = ctx.bar
    shares = ctx.holding_shares()
    rors = ctx.holding_rors()

    flat = shares <= 0.0
    if np.any(flat):
      self.pending_exit[flat] = False
      self.next_add_ror[flat] = -self.add_drop_pct
    entry_mask = flat & self.below_ma[:, cur_bar]

    holding = (~flat) & (~self.pending_exit)
    if np.any(holding):
      tp_mask = holding & (rors >= self.take_profit_pct)
      add_mask = holding & (~tp_mask) & (rors <= self.next_add_ror)

      n_tp = int(np.count_nonzero(tp_mask))
      if n_tp > 0:
        ctx.sell(tp_mask, percent=1.0)
        self.next_add_ror[tp_mask] = -self.add_drop_pct
        self.pending_exit[tp_mask] = True
        self.stats["take_profit_exits"] += n_tp
    else:
      add_mask = np.zeros(ctx.n_objs, dtype=bool)

    n_entry = int(np.count_nonzero(entry_mask))
    n_add = int(np.count_nonzero(add_mask))
    if n_entry > 0 or n_add > 0:
      buy_mask = entry_mask if n_add == 0 else (add_mask if n_entry == 0 else (entry_mask | add_mask))
      ctx.buy(buy_mask, percent=self.buy_percent)
      if n_add > 0:
        self.next_add_ror[add_mask] -= self.add_drop_pct
      self.stats["initial_entries"] += n_entry
      self.stats["add_on_buys"] += n_add

  def finalize(self, ctx: mbt.Context) -> dict:
    """Return structured execution statistics for Report diagnostics."""
    return dict(self.stats)
