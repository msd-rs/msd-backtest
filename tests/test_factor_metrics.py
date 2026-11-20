# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Unit tests for IC/RankIC, tranche quantile portfolios, turnover, holdout, and performance."""

from __future__ import annotations

import time
import numpy as np
import pytest

from mbt.factor import (
  Factor,
  FactorReport,
  MultiFactorReport,
  compute_forward_returns,
  compute_ic_series,
  compute_quantile_analysis,
  evaluate_panel,
)
from mbt.select.panel import Panel


def _make_monotone_panel(n_objs: int = 20, bars: int = 60) -> tuple[Panel, np.ndarray]:
  """Synthetic panel where stock i has deterministic daily drift proportional to i."""
  dates = np.arange(
    np.datetime64("2024-01-01"),
    np.datetime64("2024-01-01") + np.timedelta64(bars, "D"),
  )
  # Stock i grows at daily rate r_i = -0.005 + i * 0.001 (strictly increasing with i)
  rates = np.array([-0.005 + i * 0.001 for i in range(n_objs)], dtype=np.float64)
  t_idx = np.arange(bars, dtype=np.float64)[None, :]
  open_ = 10.0 * np.power(1.0 + rates[:, None], t_idx)
  close = open_ * (1.0 + 0.3 * rates[:, None])
  high = np.maximum(open_, close) * 1.005
  low = np.minimum(open_, close) * 0.995
  vol = np.full((n_objs, bars), 1e6, dtype=np.float64)
  objs = [f"SH60{i:04d}" for i in range(n_objs)]
  panel = Panel(
    {
      "ts": np.tile(dates, (n_objs, 1)),
      "open": open_,
      "high": high,
      "low": low,
      "close": close,
      "raw_open": open_,
      "raw_high": high,
      "raw_low": low,
      "raw_close": close,
      "volume": vol,
      "amount": vol * close,
      "total_shares": np.full((n_objs, bars), 1e8, dtype=np.float64),
      "limited": np.zeros((n_objs, bars), dtype=np.int32),
    },
    objs,
    bars,
  )
  # Perfect causal factor: stock index i (known at t=0, no future dependence)
  perfect_signal = np.repeat(np.arange(n_objs, dtype=np.float64)[:, None], bars, axis=1)
  return panel, perfect_signal


def test_perfect_monotone_factor_ic_and_monotonicity_and_direction():
  panel, signal = _make_monotone_panel(n_objs=20, bars=60)

  class IndexFactor(Factor):
    name = "perfect_alpha"
    def compute(self, p: Panel) -> np.ndarray:
      return signal[:, : p.bars]

  rep = evaluate_panel(
    IndexFactor(),
    panel,
    periods=(1, 5, 10),
    primary_period=5,
    quantiles=5,
    entry="next_open",
    causality_check=True,
  )
  assert isinstance(rep, FactorReport)
  assert rep.diagnostics["causality_check"]["status"] == "PASS"

  for p in (1, 5, 10):
    assert rep.ic_summaries[p].rank_ic_mean == pytest.approx(1.0, abs=1e-6)
    assert rep.ic_summaries[p].ic_mean > 0.98
    assert rep.quantile_analyses[p].monotonicity == pytest.approx(1.0, abs=1e-6)
    assert rep.quantile_analyses[p].long_short.ann_return > 0.0
    assert rep.quantile_analyses[p].long_excess_market.ann_return > 0.0

  # Negative direction flips the signal so monotonicity becomes -1 -> flipped back to +1
  class ReverseFactor(Factor):
    name = "rev_alpha"
    direction = -1
    def compute(self, p: Panel) -> np.ndarray:
      return -signal[:, : p.bars]

  rep_rev = evaluate_panel(
    ReverseFactor(),
    panel,
    periods=(1, 5),
    primary_period=5,
    quantiles=5,
    entry="next_open",
  )
  assert isinstance(rep_rev, FactorReport)
  assert rep_rev.primary_ic.rank_ic_mean == pytest.approx(1.0, abs=1e-6)
  assert rep_rev.primary_quantile.monotonicity == pytest.approx(1.0, abs=1e-6)


def test_tranche_portfolio_has_no_overlapping_compounding_inflation():
  """Constant +1%/day growth must yield ~1.01^T NAV regardless of holding period p=1, 5, or 20."""
  n_objs, bars = 15, 50
  dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + np.timedelta64(bars, "D"))
  t_idx = np.arange(bars, dtype=np.float64)[None, :]
  open_ = 10.0 * np.power(1.01, t_idx) * np.ones((n_objs, 1))
  close = open_ * 1.002
  objs = [f"SH60{i:04d}" for i in range(n_objs)]
  panel = Panel(
    {
      "ts": np.tile(dates, (n_objs, 1)),
      "open": open_,
      "high": close,
      "low": open_,
      "close": close,
      "raw_close": close,
      "volume": np.full((n_objs, bars), 1e6),
      "limited": np.zeros((n_objs, bars), dtype=np.int32),
    },
    objs,
    bars,
  )
  factor_mat = np.repeat(np.arange(n_objs, dtype=np.float64)[:, None], bars, axis=1)
  fwd_set = compute_forward_returns(panel, periods=(1, 5, 20), entry="next_open")

  qa1 = compute_quantile_analysis(factor_mat, fwd_set, 1, n_quantiles=5)
  qa5 = compute_quantile_analysis(factor_mat, fwd_set, 5, n_quantiles=5)
  qa20 = compute_quantile_analysis(factor_mat, fwd_set, 20, n_quantiles=5)

  # Active trading days for next_open open-to-open steps: bars - 2 (entry_lag=1, last bar has no next open)
  expected_nav = 1.01 ** (bars - 2)
  assert qa1.quantile_navs["Q5"][-1] == pytest.approx(expected_nav, rel=1e-5)
  assert qa5.quantile_navs["Q5"][-1] == pytest.approx(expected_nav, rel=1e-5)
  assert qa20.quantile_navs["Q5"][-1] == pytest.approx(expected_nav, rel=1e-5)


def test_entry_modes_limit_mask_cost_drag_holdout_industry_and_multifactor():
  panel, signal = _make_monotone_panel(n_objs=20, bars=80)
  # Inject limit-up on bar 10 for stock 19 and verify exclude_limit filters bar 9 signal
  limited = panel.column("limited").copy()
  limited[19, 10] = 1
  panel_lim = Panel(
    {**{c: panel.column(c) for c in panel.columns}, "limited": limited},
    panel.objs,
    panel.bars,
  )

  fwd_excl = compute_forward_returns(panel_lim, periods=(1, 5), entry="open_close", exclude_limit=True)
  fwd_incl = compute_forward_returns(panel_lim, periods=(1, 5), entry="open_close", exclude_limit=False)
  assert np.isnan(fwd_excl.forward_returns[1][19, 9])
  assert np.isfinite(fwd_incl.forward_returns[1][19, 9])

  # Close entry mode
  fwd_close = compute_forward_returns(panel_lim, periods=(5,), entry="close")
  assert np.isfinite(fwd_close.forward_returns[5][0, 0])

  # Multi-factor evaluation with holdout, cost drag, and industry breakdown
  rng = np.random.default_rng(123)
  noise = rng.normal(0.0, 3.0, size=(20, 80))

  class MultiDemo(Factor):
    def compute(self, p: Panel) -> dict[str, np.ndarray]:
      return {
        "strong": signal[:, : p.bars],
        "noisy": signal[:, : p.bars] + noise[:, : p.bars],
      }

  ind_map = {obj: ("银行" if i < 10 else "电子") for i, obj in enumerate(panel.objs)}
  rep_no_cost = evaluate_panel(
    MultiDemo(),
    panel,
    periods=(1, 5),
    primary_period=5,
    quantiles=5,
    holdout="25d",
    cost_bps=0.0,
    industry_map=ind_map,
    by_industry=True,
  )
  rep_with_cost = evaluate_panel(
    MultiDemo(),
    panel,
    periods=(1, 5),
    primary_period=5,
    quantiles=5,
    holdout="25d",
    cost_bps=25.0,
    industry_map=ind_map,
    by_industry=True,
  )

  assert isinstance(rep_no_cost, MultiFactorReport)
  assert isinstance(rep_with_cost, MultiFactorReport)
  assert rep_no_cost.correlation_matrix["strong"]["strong"] == pytest.approx(1.0)
  assert 0.3 < rep_no_cost.correlation_matrix["strong"]["noisy"] < 1.0

  noisy_0 = next(r for r in rep_no_cost.reports if r.name == "noisy")
  noisy_c = next(r for r in rep_with_cost.reports if r.name == "noisy")
  assert noisy_c.primary_quantile.quantiles[-1].total_return < noisy_0.primary_quantile.quantiles[-1].total_return
  assert noisy_0.holdout_breakdown is not None
  assert noisy_0.holdout_breakdown["in_sample"]["bars"] > 0
  assert noisy_0.holdout_breakdown["out_of_sample"]["bars"] > 0
  assert len(noisy_0.industry_breakdown) == 2


def test_factor_evaluation_performance_baseline():
  """200 stocks x 250 bars full multi-horizon evaluation + causality probe must finish < 3s."""
  rng = np.random.default_rng(99)
  n_objs, bars = 200, 250
  dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + np.timedelta64(bars, "D"))
  rets = rng.normal(0.0002, 0.02, size=(n_objs, bars))
  close = 15.0 * np.cumprod(1.0 + rets, axis=1)
  open_ = close * (1.0 + rng.normal(0.0, 0.003, size=(n_objs, bars)))
  vol = rng.uniform(1e5, 5e6, size=(n_objs, bars))
  panel = Panel(
    {
      "ts": np.tile(dates, (n_objs, 1)),
      "open": open_,
      "high": np.maximum(open_, close) * 1.01,
      "low": np.minimum(open_, close) * 0.99,
      "close": close,
      "raw_close": close,
      "volume": vol,
      "total_shares": np.full((n_objs, bars), 1e8),
      "limited": np.zeros((n_objs, bars), dtype=np.int32),
    },
    [f"SH60{i:04d}" for i in range(n_objs)],
    bars,
  )

  class MomFactor(Factor):
    name = "mom20"
    def compute(self, p: Panel) -> np.ndarray:
      c = p.column("close")
      out = np.full_like(c, np.nan)
      if p.bars > 20:
        out[:, 20:] = c[:, 20:] / c[:, :-20] - 1.0
      return out

  t0 = time.perf_counter()
  rep = evaluate_panel(
    MomFactor(),
    panel,
    periods=(1, 5, 10, 20),
    quantiles=5,
    winsorize="mad",
    neutralize=("mktcap",),
    causality_check=True,
  )
  elapsed = time.perf_counter() - t0
  assert isinstance(rep, FactorReport)
  assert elapsed < 3.0, f"Factor evaluation took too long: {elapsed:.2f}s"
