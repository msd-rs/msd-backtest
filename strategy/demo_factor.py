# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Reference multi-factor example for ``mbt.factor.cli``.

Demonstrates Agent-First factor evaluation best practices:
  * Forward-adjusted ``close`` for momentum / reversal factors.
  * Unadjusted ``raw_close`` for fundamental valuation ratio (BP = f003 / raw_close).
  * Explicit ``fin_fields`` and ``warmup_bars`` declarations.
  * Returning a multi-factor dictionary for leaderboard & correlation comparison.

Example CLI usage:
  uv run python -m mbt.factor.cli strategy/demo_factor.py -s A_STOCKS_EXCLUDE_ST -b 2022-01-01 -e 2025-12-31 -S
"""

from __future__ import annotations

import alpha as al
import numpy as np

from mbt.factor import Factor
from mbt.select import Panel

M_AUTHOR = "Antigravity"


class DemoMultiFactor(Factor):
  """Three-factor reference family: 20d reversal, 60d momentum, and book-to-price (BP)."""

  fin_fields: list[str] | None = ["f003"]
  fin_only_year: bool = False
  warmup_bars: int = 60

  def __init__(self, rev_window: int = 20, mom_window: int = 60):
    self.rev_window = max(2, int(rev_window))
    self.mom_window = max(5, int(mom_window))

  def compute(self, panel: Panel) -> dict[str, np.ndarray]:
    close = np.asarray(panel.column("close"), dtype=np.float64)
    ref_rev = panel.compute(al.REF, "close", self.rev_window)
    ref_mom = panel.compute(al.REF, "close", self.mom_window)

    with np.errstate(divide="ignore", invalid="ignore"):
      rev = np.where(ref_rev > 0.0, -(close - ref_rev) / ref_rev, np.nan)
      mom = np.where(ref_mom > 0.0, (close - ref_mom) / ref_mom, np.nan)

      raw_close = np.asarray(panel.column("raw_close"), dtype=np.float64)
      bvps = np.asarray(panel.column("f003"), dtype=np.float64)
      bp = np.where((raw_close > 0.0) & (bvps > 0.0), bvps / raw_close, np.nan)

    return {
      f"rev_{self.rev_window}d": rev,
      f"mom_{self.mom_window}d": mom,
      "value_bp": bp,
    }
