# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Vectorized factor evaluation metrics: IC/RankIC, tranche quantile returns, turnover, and breakdowns."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import alpha as al
import numpy as np

from mbt.factor.returns import ForwardReturnSet
from mbt.run.msd import find_date_index

TRADING_DAYS_PER_YEAR = 252.0


@dataclasses.dataclass
class ICSummary:
  """Aggregated IC and RankIC statistics for a single holding period."""

  period: int
  valid_bars: int
  ic_mean: float
  ic_std: float
  icir: float
  ic_positive_ratio: float
  rank_ic_mean: float
  rank_ic_std: float
  rank_icir: float
  ann_rank_icir: float
  rank_ic_positive_ratio: float
  rank_ic_sig_ratio: float
  t_stat: float
  nw_t_stat: float

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


@dataclasses.dataclass
class QuantileMetrics:
  """Return and risk statistics for a single quantile bucket or spread portfolio."""

  label: str
  mean_period_ret: float
  ann_return: float
  ann_vol: float
  sharpe: float
  max_drawdown: float
  total_return: float
  daily_turnover: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


@dataclasses.dataclass
class QuantileAnalysis:
  """Complete quantile stratification and tranche portfolio evaluation for one period."""

  period: int
  n_quantiles: int
  quantiles: list[QuantileMetrics]
  long_short: QuantileMetrics
  long_excess_market: QuantileMetrics
  long_excess_benchmark: QuantileMetrics | None
  monotonicity: float
  quantile_navs: dict[str, np.ndarray]
  long_short_nav: np.ndarray
  long_excess_nav: np.ndarray
  market_nav: np.ndarray
  benchmark_nav: np.ndarray | None
  cost_bps: float = 0.0

  def to_dict(self, include_series: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
      "period": self.period,
      "n_quantiles": self.n_quantiles,
      "monotonicity": self.monotonicity,
      "cost_bps": self.cost_bps,
      "quantiles": [q.to_dict() for q in self.quantiles],
      "long_short": self.long_short.to_dict(),
      "long_excess_market": self.long_excess_market.to_dict(),
      "long_excess_benchmark": (
        self.long_excess_benchmark.to_dict() if self.long_excess_benchmark is not None else None
      ),
    }
    if include_series:
      d["quantile_navs"] = {k: v.tolist() for k, v in self.quantile_navs.items()}
      d["long_short_nav"] = self.long_short_nav.tolist()
      d["long_excess_nav"] = self.long_excess_nav.tolist()
      d["market_nav"] = self.market_nav.tolist()
      if self.benchmark_nav is not None:
        d["benchmark_nav"] = self.benchmark_nav.tolist()
    return d


def compute_ic_series(
  factor: np.ndarray,
  fwd_ret: np.ndarray,
  *,
  min_obs: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
  """Compute daily cross-sectional Pearson IC and Spearman RankIC series.

  Uses Rust-accelerated ``alpha.CC_IC`` on mutually finite observations per bar
  with average-rank tie breaking.

  Args:
    factor: 2D factor matrix of shape ``(n_objs, bars)``.
    fwd_ret: 2D forward return matrix of shape ``(n_objs, bars)``.
    min_obs: Minimum number of valid paired observations required per bar.

  Returns:
    Tuple ``(ic_series, rank_ic_series)``, each of shape ``(bars,)``.
  """
  ic, rank_ic = al.CC_IC(factor, fwd_ret, min_obs=int(min_obs))
  return np.asarray(ic, dtype=np.float64), np.asarray(rank_ic, dtype=np.float64)


def compute_rank_ic_series(
  factor: np.ndarray,
  fwd_ret: np.ndarray,
  *,
  min_obs: int = 10,
  lag: int = 0,
) -> np.ndarray:
  """Compute daily cross-sectional Spearman RankIC series via ``alpha.CC_SPEARMAN``."""
  ric = al.CC_SPEARMAN(factor, fwd_ret, min_obs=int(min_obs), lag=int(lag))
  return np.asarray(ric, dtype=np.float64)


def _newey_west_t_stat(series: np.ndarray, max_lag: int) -> float:
  """Compute Newey-West HAC t-statistic for the mean of a 1D series."""
  v = series[np.isfinite(series)]
  n = len(v)
  if n < 3:
    return 0.0
  mu = float(np.mean(v))
  dm = v - mu
  gamma0 = float(np.dot(dm, dm) / n)
  lrv = gamma0
  lags = min(max(0, int(max_lag)), n - 2)
  for lag in range(1, lags + 1):
    w = 1.0 - lag / (lags + 1.0)
    gamma_l = float(np.dot(dm[lag:], dm[:-lag]) / n)
    lrv += 2.0 * w * gamma_l
  if lrv <= 1e-18:
    return 0.0
  se = float(np.sqrt(lrv / n))
  return mu / se if se > 0.0 else 0.0


def summarize_ic(
  ic_series: np.ndarray,
  rank_ic_series: np.ndarray,
  period: int = 1,
) -> ICSummary:
  """Compute summary statistics from daily IC and RankIC time series."""
  v_ic = ic_series[np.isfinite(ic_series)]
  v_ric = rank_ic_series[np.isfinite(rank_ic_series)]
  n = len(v_ric)

  if len(v_ic) > 0:
    ic_mean = float(np.mean(v_ic))
    ic_std = float(np.std(v_ic, ddof=1)) if len(v_ic) > 1 else 0.0
    icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
    ic_pos = float(np.mean(v_ic > 0.0))
  else:
    ic_mean = ic_std = icir = ic_pos = 0.0

  if n > 0:
    ric_mean = float(np.mean(v_ric))
    ric_std = float(np.std(v_ric, ddof=1)) if n > 1 else 0.0
    ricir = ric_mean / ric_std if ric_std > 1e-12 else 0.0
    ann_ricir = ricir * float(np.sqrt(TRADING_DAYS_PER_YEAR))
    ric_pos = float(np.mean(v_ric > 0.0))
    ric_sig = float(np.mean(np.abs(v_ric) > 0.02))
    t_stat = ric_mean / (ric_std / float(np.sqrt(n))) if ric_std > 1e-12 else 0.0
    nw_t = _newey_west_t_stat(v_ric, max_lag=max(0, int(period) - 1))
  else:
    ric_mean = ric_std = ricir = ann_ricir = ric_pos = ric_sig = t_stat = nw_t = 0.0

  return ICSummary(
    period=int(period),
    valid_bars=int(n),
    ic_mean=ic_mean,
    ic_std=ic_std,
    icir=icir,
    ic_positive_ratio=ic_pos,
    rank_ic_mean=ric_mean,
    rank_ic_std=ric_std,
    rank_icir=ricir,
    ann_rank_icir=ann_ricir,
    rank_ic_positive_ratio=ric_pos,
    rank_ic_sig_ratio=ric_sig,
    t_stat=t_stat,
    nw_t_stat=nw_t,
  )


def assign_quantiles(factor: np.ndarray, n_quantiles: int = 5) -> np.ndarray:
  """Assign each valid cross-sectional observation into ``0 .. n_quantiles - 1`` bins."""
  if n_quantiles < 2:
    raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
  n_objs, bars = factor.shape
  flat = np.ascontiguousarray(factor, dtype=np.float64).reshape(-1)
  with al.Context(groups=n_objs, flags=al.FLAG_SKIP_NAN, start=0, end=bars):
    bins = np.asarray(al.BINS(flat, int(n_quantiles)), dtype=np.float64).reshape(n_objs, bars)

  # Exclude bars with fewer than n_quantiles valid observations
  valid_cnt = np.sum(np.isfinite(factor), axis=0)
  bins[:, valid_cnt < n_quantiles] = np.nan
  return bins


def _portfolio_stats_from_daily(
  daily_ret: np.ndarray,
  *,
  label: str,
  mean_period_ret: float,
  daily_turnover: float = 0.0,
) -> tuple[QuantileMetrics, np.ndarray]:
  """Compute annualized return, vol, Sharpe, max drawdown, and NAV from daily returns."""
  valid = np.isfinite(daily_ret)
  clean = np.where(valid, daily_ret, 0.0)
  # Guard against -100% wiping NAV negative
  clean_clipped = np.maximum(clean, -0.9999)
  nav = np.cumprod(1.0 + clean_clipped)
  n_active = int(np.sum(valid))

  if n_active == 0:
    return (
      QuantileMetrics(
        label=label,
        mean_period_ret=mean_period_ret,
        ann_return=0.0,
        ann_vol=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        total_return=0.0,
        daily_turnover=daily_turnover,
      ),
      nav,
    )

  total_ret = float(nav[-1] - 1.0)
  years = n_active / TRADING_DAYS_PER_YEAR
  ann_ret = float(nav[-1] ** (1.0 / years) - 1.0) if (years > 0.0 and nav[-1] > 0.0) else -1.0

  active_rets = daily_ret[valid]
  daily_vol = float(np.std(active_rets, ddof=1)) if n_active > 1 else 0.0
  ann_vol = daily_vol * float(np.sqrt(TRADING_DAYS_PER_YEAR))
  daily_mean = float(np.mean(active_rets))
  sharpe = (
    (daily_mean / daily_vol) * float(np.sqrt(TRADING_DAYS_PER_YEAR))
    if daily_vol > 1e-12
    else 0.0
  )

  running_max = np.maximum.accumulate(np.concatenate(([1.0], nav)))[1:]
  dd = nav / running_max - 1.0
  max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

  return (
    QuantileMetrics(
      label=label,
      mean_period_ret=float(mean_period_ret),
      ann_return=ann_ret,
      ann_vol=ann_vol,
      sharpe=sharpe,
      max_drawdown=max_dd,
      total_return=total_ret,
      daily_turnover=float(daily_turnover),
    ),
    nav,
  )


def _spearman_1d(a: np.ndarray, b: np.ndarray) -> float:
  """Spearman rank correlation between two small 1D vectors (e.g. quantile means vs rank)."""
  valid = np.isfinite(a) & np.isfinite(b)
  if int(np.sum(valid)) < 2:
    return 0.0
  av = a[valid]
  bv = b[valid]

  def _avg_ranks(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    sorted_vals = arr[order]
    i = 0
    n = len(arr)
    while i < n:
      j = i + 1
      while j < n and sorted_vals[j] == sorted_vals[i]:
        j += 1
      avg_r = 0.5 * (i + j - 1)
      ranks[order[i:j]] = avg_r
      i = j
    return ranks

  ra = _avg_ranks(av)
  rb = _avg_ranks(bv)
  ra -= np.mean(ra)
  rb -= np.mean(rb)
  den = float(np.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))
  if den <= 1e-12:
    return 0.0
  return float(np.dot(ra, rb) / den)


def compute_quantile_analyses(
  factor: np.ndarray,
  fwd_set: ForwardReturnSet,
  periods: Sequence[int],
  *,
  n_quantiles: int = 5,
  cost_bps: float = 0.0,
  benchmark_daily_ret: np.ndarray | None = None,
  bins: np.ndarray | None = None,
) -> dict[int, QuantileAnalysis]:
  """Evaluate stratified quantile performance across multiple holding periods in one pass.

  Simulates ``p`` overlapping equal-weight tranches via ``alpha.TRANCHE_RETURNS``
  alongside ``alpha.BUCKET_MEANS`` and ``alpha.BUCKET_TURNOVER`` so all periods
  and quantile buckets are aggregated simultaneously in Rust.
  """
  if n_quantiles < 2:
    raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
  p_list = [int(p) for p in periods]
  if not p_list:
    return {}

  n_objs, bars = factor.shape
  bins_arr = assign_quantiles(factor, n_quantiles=n_quantiles) if bins is None else bins

  buy_mask = fwd_set.buy_mask
  entry_mode = fwd_set.entry
  if entry_mode in ("next_open", "open_close"):
    entry_ok_at_s = np.zeros((n_objs, bars), dtype=bool)
    if bars > 1:
      entry_ok_at_s[:, :-1] = buy_mask[:, 1:]
  else:
    entry_ok_at_s = buy_mask

  tranche_all = al.TRANCHE_RETURNS(
    bins_arr,
    fwd_set.step_first,
    fwd_set.step_cont,
    p_list,
    n_quantiles,
    entry_lag=fwd_set.entry_lag,
    entry_mask=entry_ok_at_s,
  )

  has_bench = benchmark_daily_ret is not None and len(benchmark_daily_ret) == bars
  b_ret: np.ndarray | None = None
  bench_nav_cached: np.ndarray | None = None
  if has_bench:
    b_ret = np.asarray(benchmark_daily_ret, dtype=np.float64)
    _, bench_nav_cached = _portfolio_stats_from_daily(
      b_ret, label="Benchmark", mean_period_ret=0.0
    )

  q_ranks = np.arange(1, n_quantiles + 1, dtype=np.float64)
  results: dict[int, QuantileAnalysis] = {}

  for pi, period in enumerate(p_list):
    fwd_ret = fwd_set.forward_returns[period]
    mkt_daily = tranche_all[pi, n_quantiles]
    _, market_nav = _portfolio_stats_from_daily(
      mkt_daily, label="Market", mean_period_ret=0.0
    )

    b_means = al.BUCKET_MEANS(bins_arr, fwd_ret, n_quantiles)
    period_turns = al.BUCKET_TURNOVER(
      bins_arr, n_quantiles, lag=period, entry_mask=entry_ok_at_s
    )

    q_daily_rets: list[np.ndarray] = []
    q_mean_fwd: list[float] = []
    q_turnovers: list[float] = []
    q_metrics: list[QuantileMetrics] = []
    q_navs: dict[str, np.ndarray] = {}

    for q in range(n_quantiles):
      label = f"Q{q + 1}"
      row_mean = b_means[q]
      fin_q = np.isfinite(row_mean)
      mean_fwd = float(np.mean(row_mean[fin_q])) if np.any(fin_q) else 0.0
      q_mean_fwd.append(mean_fwd)

      daily_turn = float(period_turns[q]) / float(period)
      q_turnovers.append(daily_turn)

      daily_r = tranche_all[pi, q]
      if cost_bps > 0.0:
        drag = 2.0 * daily_turn * float(cost_bps) * 1e-4
        daily_r = np.where(np.isfinite(daily_r), daily_r - drag, np.nan)
      else:
        daily_r = np.array(daily_r, dtype=np.float64, copy=True)
      q_daily_rets.append(daily_r)

      qm, qnav = _portfolio_stats_from_daily(
        daily_r,
        label=label,
        mean_period_ret=mean_fwd,
        daily_turnover=daily_turn,
      )
      q_metrics.append(qm)
      q_navs[label] = qnav

    top_daily = q_daily_rets[-1]
    bot_daily = q_daily_rets[0]

    ls_daily = np.where(
      np.isfinite(top_daily) & np.isfinite(bot_daily),
      top_daily - bot_daily,
      np.nan,
    )
    ls_metrics, ls_nav = _portfolio_stats_from_daily(
      ls_daily,
      label=f"Q{n_quantiles}-Q1",
      mean_period_ret=q_mean_fwd[-1] - q_mean_fwd[0],
      daily_turnover=q_turnovers[-1] + q_turnovers[0],
    )

    row_mkt_mean = b_means[n_quantiles]
    fin_m = np.isfinite(row_mkt_mean)
    mkt_mean_fwd = float(np.mean(row_mkt_mean[fin_m])) if np.any(fin_m) else 0.0

    lex_daily = np.where(
      np.isfinite(top_daily) & np.isfinite(mkt_daily),
      top_daily - mkt_daily,
      np.nan,
    )
    lex_metrics, lex_nav = _portfolio_stats_from_daily(
      lex_daily,
      label=f"Q{n_quantiles}-Mkt",
      mean_period_ret=q_mean_fwd[-1] - mkt_mean_fwd,
      daily_turnover=q_turnovers[-1],
    )

    lb_metrics: QuantileMetrics | None = None
    bench_nav: np.ndarray | None = None
    if b_ret is not None and bench_nav_cached is not None:
      bench_nav = np.array(bench_nav_cached, dtype=np.float64, copy=True)
      lb_daily = np.where(
        np.isfinite(top_daily) & np.isfinite(b_ret),
        top_daily - b_ret,
        np.nan,
      )
      lb_metrics, _ = _portfolio_stats_from_daily(
        lb_daily,
        label=f"Q{n_quantiles}-Bench",
        mean_period_ret=0.0,
        daily_turnover=q_turnovers[-1],
      )

    mono = _spearman_1d(q_ranks, np.asarray(q_mean_fwd, dtype=np.float64))

    results[period] = QuantileAnalysis(
      period=period,
      n_quantiles=int(n_quantiles),
      quantiles=q_metrics,
      long_short=ls_metrics,
      long_excess_market=lex_metrics,
      long_excess_benchmark=lb_metrics,
      monotonicity=mono,
      quantile_navs=q_navs,
      long_short_nav=ls_nav,
      long_excess_nav=lex_nav,
      market_nav=market_nav,
      benchmark_nav=bench_nav,
      cost_bps=float(cost_bps),
    )

  return results


def compute_quantile_analysis(
  factor: np.ndarray,
  fwd_set: ForwardReturnSet,
  period: int,
  *,
  n_quantiles: int = 5,
  cost_bps: float = 0.0,
  benchmark_daily_ret: np.ndarray | None = None,
  bins: np.ndarray | None = None,
) -> QuantileAnalysis:
  """Evaluate stratified quantile performance using non-inflating tranche portfolios.

  For holding period ``p``, simulates ``p`` overlapping equal-weight tranches
  where ``1/p`` of capital is rebalanced daily into the latest quantile bins and
  held for ``p`` trading days. This eliminates overlapping-return compounding
  inflation while removing rebalance-day start-date bias.

  Args:
    factor: Preprocessed & direction-aligned factor matrix ``(n_objs, bars)``.
    fwd_set: Precomputed :class:`ForwardReturnSet`.
    period: Holding period present in ``fwd_set.forward_returns``.
    n_quantiles: Number of quantile groups (e.g. 5 or 10).
    cost_bps: One-way transaction cost (commission + slippage + half-tax) in bps.
    benchmark_daily_ret: Optional 1D benchmark daily return array of shape ``(bars,)``.
    bins: Optional precomputed quantile bin matrix from :func:`assign_quantiles`.

  Returns:
    :class:`QuantileAnalysis` with per-quantile metrics, Long-Short, Long-Excess,
    monotonicity, and NAV time series.
  """
  p = int(period)
  return compute_quantile_analyses(
    factor,
    fwd_set,
    (p,),
    n_quantiles=n_quantiles,
    cost_bps=cost_bps,
    benchmark_daily_ret=benchmark_daily_ret,
    bins=bins,
  )[p]


def compute_turnover_and_autocorr(
  factor: np.ndarray,
  *,
  n_quantiles: int = 5,
  lags: Sequence[int] = (1, 5, 10, 20),
  min_obs: int = 10,
  bins: np.ndarray | None = None,
) -> dict[str, Any]:
  """Compute top/bottom quantile daily turnover, rank autocorrelation, and half-life."""
  bars = factor.shape[1]
  bins_arr = assign_quantiles(factor, n_quantiles=n_quantiles) if bins is None else bins
  turns_1d = al.BUCKET_TURNOVER(bins_arr, n_quantiles, lag=1)
  top_turn_1d = float(turns_1d[n_quantiles - 1])
  bot_turn_1d = float(turns_1d[0])

  autocorr: dict[int, float] = {}
  for lag in lags:
    lag_i = int(lag)
    if lag_i <= 0 or bars <= lag_i:
      continue
    ric = compute_rank_ic_series(factor, factor, min_obs=min_obs, lag=lag_i)
    v = ric[np.isfinite(ric)]
    autocorr[lag_i] = float(np.mean(v)) if len(v) > 0 else float("nan")

  half_life: float | None = None
  rho1 = autocorr.get(1)
  if rho1 is not None and np.isfinite(rho1) and 0.0 < rho1 < 0.999999:
    half_life = float(-np.log(2.0) / np.log(rho1))

  return {
    "top_daily_turnover": top_turn_1d,
    "bottom_daily_turnover": bot_turn_1d,
    "autocorrelation": autocorr,
    "half_life_days": half_life,
  }


def compute_annual_breakdown(
  dates: np.ndarray,
  ic_series: np.ndarray,
  rank_ic_series: np.ndarray,
  long_excess_nav: np.ndarray,
  period: int = 1,
) -> list[dict[str, Any]]:
  """Compute per-calendar-year IC/RankIC and Top-Market excess return breakdown."""
  n = len(dates)
  if n == 0:
    return []

  years = [str(d)[:4] for d in dates]
  unique_years = sorted(set(years))
  rows: list[dict[str, Any]] = []

  for yr in unique_years:
    idx = np.array([i for i, y in enumerate(years) if y == yr], dtype=np.int64)
    if len(idx) == 0:
      continue
    ic_sum = summarize_ic(ic_series[idx], rank_ic_series[idx], period=period)
    # Year excess return from cumulative NAV
    i0, i1 = int(idx[0]), int(idx[-1])
    base_nav = float(long_excess_nav[i0 - 1]) if i0 > 0 else 1.0
    end_nav = float(long_excess_nav[i1])
    yr_excess = (end_nav / base_nav - 1.0) if base_nav > 0.0 else 0.0

    rows.append({
      "year": yr,
      "bars": len(idx),
      "valid_ic_bars": ic_sum.valid_bars,
      "ic_mean": ic_sum.ic_mean,
      "rank_ic_mean": ic_sum.rank_ic_mean,
      "rank_icir": ic_sum.rank_icir,
      "rank_ic_positive_ratio": ic_sum.rank_ic_positive_ratio,
      "top_excess_return": yr_excess,
    })
  return rows


def _resolve_holdout_index(dates: np.ndarray, holdout: str | int) -> int:
  """Resolve a holdout date (YYYY-MM-DD) or trailing duration ('60d', '1y') to a bar index."""
  n = len(dates)
  if n < 4:
    raise ValueError(f"Insufficient bars ({n}) for holdout split '{holdout}'.")

  h_str = str(holdout).strip()
  try:
    if h_str.lower().endswith("d"):
      days = int(h_str[:-1])
      split_idx = n - days
    elif h_str.lower().endswith("y"):
      days = int(round(float(h_str[:-1]) * 250))
      split_idx = n - days
    else:
      split_idx = find_date_index(dates, h_str)
  except Exception as e:
    raise ValueError(
      f"Invalid --holdout value '{holdout}'. Expected a date string (YYYY-MM-DD), "
      "bar count (e.g. '60d'), or year duration (e.g. '1y')."
    ) from e

  if split_idx <= 1 or split_idx >= n - 1:
    start_s = str(dates[0])[:10]
    end_s = str(dates[-1])[:10]
    raise ValueError(
      f"Holdout '{holdout}' resolves to split_index={split_idx}, which leaves an empty "
      f"in-sample or out-of-sample segment within [{start_s}, {end_s}] ({n} bars)."
    )
  return int(split_idx)


def compute_holdout_breakdown(
  dates: np.ndarray,
  factor: np.ndarray,
  fwd_set: ForwardReturnSet,
  primary_period: int,
  holdout: str,
  *,
  n_quantiles: int = 5,
  cost_bps: float = 0.0,
  min_obs: int = 10,
) -> dict[str, Any]:
  """Compare In-Sample (IS) vs Out-of-Sample (OOS) factor performance at ``holdout``."""
  split_idx = _resolve_holdout_index(dates, holdout)
  split_date = str(dates[split_idx])[:10]

  def _eval_slice(sl: slice) -> dict[str, Any]:
    f_sub = factor[:, sl]
    r_sub = fwd_set.forward_returns[primary_period][:, sl]
    ic_s, ric_s = compute_ic_series(f_sub, r_sub, min_obs=min_obs)
    ic_stat = summarize_ic(ic_s, ric_s, period=primary_period)

    sub_fwd_set = ForwardReturnSet(
      periods=(primary_period,),
      entry=fwd_set.entry,
      forward_returns={primary_period: r_sub},
      signal_mask=fwd_set.signal_mask[:, sl],
      buy_mask=fwd_set.buy_mask[:, sl],
      step_first=fwd_set.step_first[:, sl],
      step_cont=fwd_set.step_cont[:, sl],
      entry_lag=fwd_set.entry_lag,
      suspended_ratio=fwd_set.suspended_ratio,
      limit_filtered_ratio=fwd_set.limit_filtered_ratio,
    )
    qa = compute_quantile_analysis(
      f_sub,
      sub_fwd_set,
      primary_period,
      n_quantiles=n_quantiles,
      cost_bps=cost_bps,
    )
    sub_dates = dates[sl]
    return {
      "start": str(sub_dates[0])[:10] if len(sub_dates) > 0 else "",
      "end": str(sub_dates[-1])[:10] if len(sub_dates) > 0 else "",
      "bars": int(len(sub_dates)),
      "rank_ic_mean": ic_stat.rank_ic_mean,
      "rank_icir": ic_stat.rank_icir,
      "ic_mean": ic_stat.ic_mean,
      "top_ann_excess": qa.long_excess_market.ann_return,
      "ls_ann_return": qa.long_short.ann_return,
      "monotonicity": qa.monotonicity,
    }

  is_stats = _eval_slice(slice(0, split_idx))
  oos_stats = _eval_slice(slice(split_idx, len(dates)))

  retention: float | None = None
  if abs(is_stats["rank_ic_mean"]) > 1e-6:
    retention = float(oos_stats["rank_ic_mean"] / is_stats["rank_ic_mean"])

  return {
    "holdout_arg": holdout,
    "split_date": split_date,
    "split_index": int(split_idx),
    "period": int(primary_period),
    "in_sample": is_stats,
    "out_of_sample": oos_stats,
    "rank_ic_retention": retention,
  }


def compute_industry_breakdown(
  factor: np.ndarray,
  fwd_ret: np.ndarray,
  industry_names: Sequence[str | None],
  *,
  min_industry_obs: int = 3,
  period: int = 1,
) -> list[dict[str, Any]]:
  """Compute average RankIC and coverage per industry."""
  groups: dict[str, list[int]] = {}
  for idx, ind in enumerate(industry_names):
    if ind:
      groups.setdefault(ind, []).append(idx)

  rows: list[dict[str, Any]] = []
  for ind_name, row_indices in sorted(groups.items()):
    if len(row_indices) < min_industry_obs:
      continue
    idx_arr = np.asarray(row_indices, dtype=np.int64)
    f_sub = factor[idx_arr, :]
    r_sub = fwd_ret[idx_arr, :]
    ic_s, ric_s = compute_ic_series(f_sub, r_sub, min_obs=min_industry_obs)
    summary = summarize_ic(ic_s, ric_s, period=period)
    valid_cnt = np.sum(np.isfinite(f_sub) & np.isfinite(r_sub), axis=0)
    rows.append({
      "industry": ind_name,
      "n_stocks": len(row_indices),
      "avg_valid_stocks": float(np.mean(valid_cnt)),
      "valid_bars": summary.valid_bars,
      "rank_ic_mean": summary.rank_ic_mean,
      "rank_icir": summary.rank_icir,
      "ic_mean": summary.ic_mean,
    })

  rows.sort(key=lambda r: r["rank_ic_mean"], reverse=True)
  return rows


def compute_factor_correlation_matrix(
  factors: dict[str, np.ndarray],
  *,
  min_obs: int = 10,
) -> dict[str, dict[str, float]]:
  """Compute pairwise average cross-sectional Spearman rank correlation matrix."""
  names = list(factors.keys())
  if not names:
    return {}
  corr_arr = al.CC_SPEARMAN_MATRIX([factors[n] for n in names], min_obs=int(min_obs))
  return {
    ni: {nj: float(corr_arr[i, j]) for j, nj in enumerate(names)}
    for i, ni in enumerate(names)
  }
