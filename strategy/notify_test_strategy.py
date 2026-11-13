# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Deterministic test portfolio strategy guaranteed to emit trade signals on every bar.

Designed for end-to-end testing of PortfolioAccount execution and webhook
notifications (`--notify`, `--notify-on-trade`, `--exec-mode close|next_open`).

Usage:
  uv run python -m mbt.run.cli strategy/notify_test_strategy.py \\
    -s SH600000 SZ000001 -p -b 2026-01-01 -S --notify etc/notify.json --notify-on-trade
"""

from __future__ import annotations

import numpy as np
import mbt


class NotifyTestPortfolioStrategy(mbt.PortfolioStrategy):
  """Alternating-weight portfolio strategy that always triggers rebalancing."""

  portfolio: bool = True

  def __init__(self, high_weight: float = 0.8, low_weight: float = 0.2):
    super().__init__()
    self.high_weight = float(high_weight)
    self.low_weight = float(low_weight)

  def execute(self, ctx: mbt.PortfolioContext):
    tradable_idx = np.flatnonzero(ctx.account.tradable_mask)
    if len(tradable_idx) == 0:
      return

    weights = np.zeros(ctx.dp.groups, dtype=np.float64)
    is_even = (ctx.bar % 2) == 0

    if len(tradable_idx) == 1:
      weights[tradable_idx[0]] = self.high_weight if is_even else self.low_weight
    else:
      primary = tradable_idx[0] if is_even else tradable_idx[1]
      secondary = tradable_idx[1] if is_even else tradable_idx[0]
      weights[primary] = self.high_weight
      weights[secondary] = self.low_weight

    ctx.rebalance(weights, tolerance=0.0)
