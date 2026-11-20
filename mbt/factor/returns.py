# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Forward return computation and A-share tradability masks for factor evaluation."""

from __future__ import annotations

import dataclasses
from typing import Sequence

import alpha as al
import numpy as np

from mbt.select.panel import Panel

VALID_ENTRY_MODES = ("next_open", "open_close", "close")


@dataclasses.dataclass
class ForwardReturnSet:
  """Precomputed forward returns, tradability masks, and daily step returns.

  All matrices have shape ``(n_objs, bars)`` aligned with the evaluation ``Panel``.
  """

  periods: tuple[int, ...]
  entry: str
  forward_returns: dict[int, np.ndarray]
  signal_mask: np.ndarray
  buy_mask: np.ndarray
  # Step return arrays used by tranche portfolio simulation:
  # - step_first[:, d]: return on entry bar d (for entry="next_open" this is open[d+1]/open[d]-1;
  #   for "open_close" this is close[d]/open[d]-1; for "close" this is close[d]/close[d-1]-1)
  # - step_cont[:, d]: return on continuation bar d
  step_first: np.ndarray
  step_cont: np.ndarray
  entry_lag: int
  suspended_ratio: float
  limit_filtered_ratio: float


def build_tradability_masks(
  panel: Panel,
  *,
  exclude_limit: bool = True,
  min_listed_days: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
  """Build boolean ``(n_objs, bars)`` signal and entry-tradability masks.

  Args:
    panel: Rectangular ``(n_objs, bars)`` market panel.
    exclude_limit: When ``True``, bars where ``limited == 1`` (limit-up) are
      marked non-buyable.
    min_listed_days: Minimum number of valid trading bars since a stock's first
      valid bar in the panel before it becomes eligible for factor signals.

  Returns:
    Tuple ``(signal_mask, buy_mask, suspended_ratio, limit_filtered_ratio)``.
  """
  n_objs, bars = panel.n_objs, panel.bars
  close = np.asarray(panel.column("close"), dtype=np.float64)
  open_ = np.asarray(panel.column("open"), dtype=np.float64)

  valid_price = np.isfinite(close) & (close > 0.0) & np.isfinite(open_) & (open_ > 0.0)
  if "volume" in panel.columns:
    vol = np.asarray(panel.column("volume"), dtype=np.float64)
    valid_bar = valid_price & np.isfinite(vol) & (vol > 0.0)
  else:
    valid_bar = valid_price

  total_cells = max(1, n_objs * bars)
  suspended_ratio = float(1.0 - np.sum(valid_bar) / total_cells)

  signal_mask = valid_bar.copy()
  if min_listed_days > 0 and bars > 0:
    has_any = np.any(valid_bar, axis=1)
    first_valid = np.argmax(valid_bar, axis=1)
    bar_idx = np.arange(bars, dtype=np.int64)[None, :]
    listed_ok = has_any[:, None] & (bar_idx >= (first_valid[:, None] + int(min_listed_days)))
    signal_mask &= listed_ok

  buy_mask = valid_bar.copy()
  limit_filtered_ratio = 0.0
  if exclude_limit and "limited" in panel.columns:
    limited = np.asarray(panel.column("limited"))
    is_limit_up = limited == 1
    n_valid = int(np.sum(valid_bar))
    if n_valid > 0:
      limit_filtered_ratio = float(np.sum(valid_bar & is_limit_up) / n_valid)
    buy_mask &= ~is_limit_up

  return signal_mask, buy_mask, suspended_ratio, limit_filtered_ratio


def compute_forward_returns(
  panel: Panel,
  periods: Sequence[int] = (1, 5, 10, 20),
  *,
  entry: str = "next_open",
  exclude_limit: bool = True,
  min_listed_days: int = 0,
) -> ForwardReturnSet:
  """Compute forward total returns on forward-adjusted prices with tradability masks.

  Forward returns use forward-adjusted ``open`` / ``close`` prices so dividends,
  bonus shares, and splits within the holding window are accurately included in
  total return. Tradability masks use ``volume`` and unadjusted-basis ``limited``.

  Entry Alignment Modes:
    * ``"next_open"`` (default): Signal at bar ``t`` close -> enter at bar
      ``t + 1`` open -> exit at bar ``t + 1 + p`` open. Matches
      ``PortfolioStrategy(exec_mode="next_open")`` and captures full overnight
      and intraday returns over ``p`` trading days.
    * ``"open_close"``: Signal at bar ``t`` close -> enter at bar ``t + 1`` open
      -> exit at bar ``t + p`` close, computed via ``alpha.FRET``.
    * ``"close"``: Signal at bar ``t`` close -> enter at bar ``t`` close -> exit
      at bar ``t + p`` close (theoretical reference without execution lag).

  Args:
    panel: Rectangular ``(n_objs, bars)`` panel.
    periods: Positive integer holding periods in trading bars.
    entry: Execution alignment mode (``"next_open"``, ``"open_close"``, or ``"close"``).
    exclude_limit: Exclude limit-up bars (``limited == 1``) on entry day.
    min_listed_days: Minimum valid trading bars before a stock enters the universe.

  Returns:
    :class:`ForwardReturnSet` containing per-period forward return matrices and
    step-return matrices for tranche portfolio evaluation.
  """
  if entry not in VALID_ENTRY_MODES:
    raise ValueError(
      f"Invalid entry mode {entry!r}. Supported modes: {VALID_ENTRY_MODES}."
    )

  norm_periods = tuple(sorted({int(p) for p in periods}))
  if not norm_periods or any(p <= 0 for p in norm_periods):
    raise ValueError(f"periods must be non-empty positive integers, got {periods!r}.")

  n_objs, bars = panel.n_objs, panel.bars
  open_ = np.asarray(panel.column("open"), dtype=np.float64)
  close = np.asarray(panel.column("close"), dtype=np.float64)

  signal_mask, buy_mask, susp_ratio, limit_ratio = build_tradability_masks(
    panel,
    exclude_limit=exclude_limit,
    min_listed_days=min_listed_days,
  )

  fwd: dict[int, np.ndarray] = {}

  if entry == "next_open":
    # Enter at open[t+1], exit at open[t+1+p]
    for p in norm_periods:
      mat = np.full((n_objs, bars), np.nan, dtype=np.float64)
      if bars > p + 1:
        end_t = bars - 1 - p
        p_in = open_[:, 1 : 1 + end_t]
        p_out = open_[:, 1 + p : 1 + p + end_t]
        ok = (
          signal_mask[:, :end_t]
          & buy_mask[:, 1 : 1 + end_t]
          & np.isfinite(p_in) & (p_in > 0.0)
          & np.isfinite(p_out) & (p_out > 0.0)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
          ret = np.where(ok, p_out / p_in - 1.0, np.nan)
        mat[:, :end_t] = ret
      fwd[p] = mat

    # Daily open-to-open step return attributed to day d = entry_bar (open[d+1]/open[d] - 1)
    step = np.full((n_objs, bars), np.nan, dtype=np.float64)
    if bars > 1:
      p0 = open_[:, :-1]
      p1 = open_[:, 1:]
      ok = np.isfinite(p0) & (p0 > 0.0) & np.isfinite(p1) & (p1 > 0.0)
      with np.errstate(divide="ignore", invalid="ignore"):
        step[:, :-1] = np.where(ok, p1 / p0 - 1.0, np.nan)
    step_first = step
    step_cont = step
    entry_lag = 1

  elif entry == "open_close":
    # Enter at open[t+1], exit at close[t+p] via Rust-accelerated al.FRET
    open_flat = np.ascontiguousarray(open_, dtype=np.float64).reshape(-1)
    close_flat = np.ascontiguousarray(close, dtype=np.float64).reshape(-1)
    calc_flat = np.ascontiguousarray(buy_mask, dtype=np.float64).reshape(-1)
    with panel.ctx():
      for p in norm_periods:
        fret_2d = np.asarray(
          al.FRET(open_flat, close_flat, calc_flat, 1, p), dtype=np.float64
        ).reshape(n_objs, bars)
        fwd[p] = np.where(signal_mask, fret_2d, np.nan)

    # First holding day d = t+1 earns open[d] -> close[d]; subsequent days earn close[d-1] -> close[d]
    step_first = np.full((n_objs, bars), np.nan, dtype=np.float64)
    ok_oc = np.isfinite(open_) & (open_ > 0.0) & np.isfinite(close) & (close > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
      step_first = np.where(ok_oc, close / open_ - 1.0, np.nan)

    step_cont = np.full((n_objs, bars), np.nan, dtype=np.float64)
    if bars > 1:
      c0 = close[:, :-1]
      c1 = close[:, 1:]
      ok_cc = np.isfinite(c0) & (c0 > 0.0) & np.isfinite(c1) & (c1 > 0.0)
      with np.errstate(divide="ignore", invalid="ignore"):
        step_cont[:, 1:] = np.where(ok_cc, c1 / c0 - 1.0, np.nan)
    entry_lag = 1

  else:
    # entry == "close": signal at bar t close -> buy at close[t] -> sell at close[t+p]
    for p in norm_periods:
      mat = np.full((n_objs, bars), np.nan, dtype=np.float64)
      if bars > p:
        end_t = bars - p
        p_in = close[:, :end_t]
        p_out = close[:, p : p + end_t]
        ok = (
          signal_mask[:, :end_t]
          & buy_mask[:, :end_t]
          & np.isfinite(p_in) & (p_in > 0.0)
          & np.isfinite(p_out) & (p_out > 0.0)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
          ret = np.where(ok, p_out / p_in - 1.0, np.nan)
        mat[:, :end_t] = ret
      fwd[p] = mat

    step_cc = np.full((n_objs, bars), np.nan, dtype=np.float64)
    if bars > 1:
      c0 = close[:, :-1]
      c1 = close[:, 1:]
      ok_cc = np.isfinite(c0) & (c0 > 0.0) & np.isfinite(c1) & (c1 > 0.0)
      with np.errstate(divide="ignore", invalid="ignore"):
        step_cc[:, 1:] = np.where(ok_cc, c1 / c0 - 1.0, np.nan)
    step_first = step_cc
    step_cont = step_cc
    entry_lag = 1  # first realized return bar for signal at t is bar t+1

  return ForwardReturnSet(
    periods=norm_periods,
    entry=entry,
    forward_returns=fwd,
    signal_mask=signal_mask,
    buy_mask=buy_mask,
    step_first=step_first,
    step_cont=step_cont,
    entry_lag=entry_lag,
    suspended_ratio=susp_ratio,
    limit_filtered_ratio=limit_ratio,
  )
