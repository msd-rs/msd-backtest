# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Factor evaluation orchestrator bridging MSD data loading, preprocessing, and metrics."""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import numpy as np

from mbt.factor.expr import ExpressionFactor, load_expression_file
from mbt.factor.factor import (
  Factor,
  check_factor_causality,
  load_factor,
  normalize_factor_output,
)
from mbt.factor.metrics import (
  ICSummary,
  QuantileAnalysis,
  assign_quantiles,
  compute_annual_breakdown,
  compute_factor_correlation_matrix,
  compute_holdout_breakdown,
  compute_ic_series,
  compute_industry_breakdown,
  compute_quantile_analyses,
  compute_rank_ic_series,
  compute_turnover_and_autocorr,
  summarize_ic,
)
from mbt.factor.preprocess import compute_log_mktcap, preprocess_factor
from mbt.factor.report import FactorReport, MultiFactorReport
from mbt.factor.returns import compute_forward_returns
from mbt.select.panel import Panel

logger = logging.getLogger("factor")


def _encode_industries(
  objs: Sequence[str],
  industry_map: dict[str, str] | None,
) -> tuple[list[str | None], np.ndarray | None]:
  if industry_map is None:
    return [None] * len(objs), None
  names: list[str | None] = [industry_map.get(sym) for sym in objs]
  unique = sorted({n for n in names if n is not None})
  name_to_code = {n: float(i + 1) for i, n in enumerate(unique)}
  codes = np.array(
    [name_to_code[n] if n is not None else np.nan for n in names],
    dtype=np.float64,
  )
  return names, codes


def _select_primary_period(periods: tuple[int, ...], primary_period: int | None) -> int:
  if primary_period is not None:
    p = int(primary_period)
    if p not in periods:
      raise ValueError(
        f"primary_period={p} is not in evaluated periods {periods}."
      )
    return p
  if 5 in periods:
    return 5
  return periods[len(periods) // 2]


def _benchmark_daily_returns(
  benchmark_close: np.ndarray | None,
  benchmark_open: np.ndarray | None,
  entry: str,
) -> np.ndarray | None:
  if entry == "next_open" and benchmark_open is not None:
    arr = np.asarray(benchmark_open, dtype=np.float64)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) > 1:
      p0, p1 = arr[:-1], arr[1:]
      ok = np.isfinite(p0) & (p0 > 0.0) & np.isfinite(p1) & (p1 > 0.0)
      with np.errstate(divide="ignore", invalid="ignore"):
        out[:-1] = np.where(ok, p1 / p0 - 1.0, np.nan)
    return out

  if benchmark_close is None:
    return None
  arr = np.asarray(benchmark_close, dtype=np.float64)
  out = np.full(len(arr), np.nan, dtype=np.float64)
  if len(arr) > 1:
    p0, p1 = arr[:-1], arr[1:]
    ok = np.isfinite(p0) & (p0 > 0.0) & np.isfinite(p1) & (p1 > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
      out[1:] = np.where(ok, p1 / p0 - 1.0, np.nan)
  return out


def _build_warnings(
  *,
  name: str,
  direction: int,
  primary_ic: ICSummary,
  primary_qa: QuantileAnalysis,
  coverage_mean: float,
  mktcap_corr: float | None,
  neutralize: tuple[str, ...],
  top_daily_turnover: float,
  holdout_breakdown: dict[str, Any] | None,
  constant_cs_days: int,
  bars: int,
) -> list[str]:
  alerts: list[str] = []
  if primary_ic.rank_ic_mean < -0.02:
    alerts.append(
      f"Negative primary RankIC ({primary_ic.rank_ic_mean:.4f}): factor '{name}' may be "
      f"inversely predictive; consider setting direction={-direction}."
    )
  if mktcap_corr is not None and np.isfinite(mktcap_corr) and abs(mktcap_corr) > 0.30 and "mktcap" not in neutralize:
    alerts.append(
      f"High market-cap exposure (RankCorr with log-mktcap = {mktcap_corr:.3f}). "
      "Re-test with `--neutralize industry,mktcap` to verify pure alpha."
    )
  if coverage_mean < 0.50:
    alerts.append(
      f"Low average factor coverage ({coverage_mean * 100:.1f}%). Verify missing financial/price fields."
    )
  if constant_cs_days > max(5, int(bars * 0.1)):
    alerts.append(
      f"{constant_cs_days} cross-section bars have zero variance (constant factor values)."
    )
  if abs(primary_qa.monotonicity) < 0.50 and primary_ic.valid_bars >= 20:
    alerts.append(
      f"Weak quantile monotonicity ({primary_qa.monotonicity:.3f}); returns do not scale smoothly across buckets."
    )
  if top_daily_turnover > 0.50:
    alerts.append(
      f"High top-quantile daily turnover ({top_daily_turnover * 100:.1f}%). "
      "Evaluate cost-adjusted returns with `--cost-bps`."
    )
  if holdout_breakdown is not None:
    ret = holdout_breakdown.get("rank_ic_retention")
    if ret is not None and np.isfinite(ret) and ret < 0.50:
      alerts.append(
        f"Severe out-of-sample RankIC decay (OOS/IS retention = {ret:.2f}x); potential overfitting."
      )
  return alerts


def evaluate_panel(
  factor: Factor,
  panel: Panel,
  *,
  periods: Sequence[int] = (1, 5, 10, 20),
  primary_period: int | None = None,
  quantiles: int = 5,
  entry: str = "next_open",
  exclude_limit: bool = True,
  min_listed_days: int = 0,
  winsorize: str = "none",
  standardize: str = "none",
  neutralize: Sequence[str] = (),
  cost_bps: float = 0.0,
  warmup_bars: int = 0,
  holdout: str | None = None,
  causality_check: bool = True,
  min_obs: int = 10,
  industry_map: dict[str, str] | None = None,
  by_industry: bool = False,
  benchmark_name: str = "",
  benchmark_close: np.ndarray | None = None,
  benchmark_open: np.ndarray | None = None,
) -> FactorReport | MultiFactorReport:
  """Evaluate a :class:`Factor` (or multi-factor family) directly on an in-memory ``Panel``.

  Args:
    factor: Instantiated :class:`Factor` (including :class:`ExpressionFactor`).
    panel: Rectangular ``(n_objs, bars)`` historical data panel (may include leading
      ``warmup_bars`` bars).
    periods: Holding periods in trading bars.
    primary_period: Highlighted period for quantile tables, annual/holdout breakdowns,
      and multi-factor leaderboards. Defaults to ``5`` when present.
    quantiles: Number of cross-sectional buckets (default 5).
    entry: Execution alignment (``"next_open"``, ``"open_close"``, or ``"close"``).
    exclude_limit: Exclude limit-up bars (``limited == 1``) from entry tradability.
    min_listed_days: Exclude stocks with fewer than this many prior valid bars.
    winsorize: Cross-sectional winsorization method (``"none"``, ``"mad"``, ``"sigma"``, ``"percentile"``).
    standardize: Cross-sectional standardization (``"none"``, ``"zscore"``, ``"rank"``).
    neutralize: Neutralization factors, subset of ``("industry", "mktcap")``.
    cost_bps: One-way transaction cost in basis points deducted from tranche returns.
    warmup_bars: Leading bars excluded from metrics after computing rolling factors.
    holdout: Optional holdout split date or trailing duration (e.g. ``"2024-01-01"``, ``"1y"``).
    causality_check: Run truncation causality probes to fail fast on lookahead bias.
    min_obs: Minimum valid cross-sectional stock count per bar for IC computation.
    industry_map: Optional ``{obj: industry_name}`` mapping.
    by_industry: When ``True`` and ``industry_map`` is supplied, compute per-industry RankIC.
    benchmark_name: Benchmark symbol label for reporting.
    benchmark_close: Optional 1D benchmark close series aligned to ``panel.timeline()``.
    benchmark_open: Optional 1D benchmark open series aligned to ``panel.timeline()``.

  Returns:
    :class:`FactorReport` when the factor produces one signal matrix, or
    :class:`MultiFactorReport` when it produces multiple named factors.
  """
  norm_periods = tuple(sorted({int(p) for p in periods}))
  if not norm_periods or any(p <= 0 for p in norm_periods):
    raise ValueError(f"periods must be positive integers, got {periods!r}.")
  prim_p = _select_primary_period(norm_periods, primary_period)

  norm_neut = tuple(
    dict.fromkeys(m.strip().lower() for m in neutralize if m and m.strip().lower() != "none")
  )
  w = max(0, int(warmup_bars))
  if w >= panel.bars - 1:
    raise ValueError(
      f"warmup_bars ({w}) leaves insufficient evaluation bars from total panel bars ({panel.bars})."
    )

  industry_names, industry_codes = _encode_industries(panel.objs, industry_map)

  if isinstance(factor, ExpressionFactor):
    factor.bind_context_metadata(
      industry_codes=industry_codes,
      benchmark_close=benchmark_close,
      benchmark_open=benchmark_open,
    )

  default_name = factor.name or factor.__class__.__name__
  raw_out = factor.compute(panel)
  full_outputs = normalize_factor_output(raw_out, panel.n_objs, panel.bars, default_name)

  if causality_check:
    causality_res = check_factor_causality(
      factor,
      panel,
      full_outputs,
      warmup_bars=w,
      fail_fast=True,
    )
  else:
    causality_res = {"status": "SKIPPED", "reason": "disabled by user", "probes": [], "violations": []}

  # Slice off warmup window before computing forward returns & evaluation stats
  if w > 0:
    eval_panel = panel.slice(w, panel.bars)
    eval_outputs = {k: np.ascontiguousarray(v[:, w:]) for k, v in full_outputs.items()}
    eval_b_close = benchmark_close[w:] if benchmark_close is not None else None
    eval_b_open = benchmark_open[w:] if benchmark_open is not None else None
  else:
    eval_panel = panel
    eval_outputs = full_outputs
    eval_b_close = benchmark_close
    eval_b_open = benchmark_open

  dates = eval_panel.timeline()
  start_date = str(dates[0])[:10] if len(dates) > 0 else ""
  end_date = str(dates[-1])[:10] if len(dates) > 0 else ""

  fwd_set = compute_forward_returns(
    eval_panel,
    periods=norm_periods,
    entry=entry,
    exclude_limit=exclude_limit,
    min_listed_days=min_listed_days,
  )

  # Compute log market cap whenever available (mandatory if mktcap neutralization requested)
  log_mktcap: np.ndarray | None = None
  if "mktcap" in norm_neut:
    log_mktcap = compute_log_mktcap(eval_panel)
  elif ("raw_close" in eval_panel.columns or "price" in eval_panel.columns) and "total_shares" in eval_panel.columns:
    try:
      log_mktcap = compute_log_mktcap(eval_panel)
    except Exception:
      log_mktcap = None

  bench_daily_ret = _benchmark_daily_returns(eval_b_close, eval_b_open, entry)
  custom_diag = factor.finalize()
  author = str(getattr(factor, "_author", ""))

  processed_map: dict[str, np.ndarray] = {}
  reports: list[FactorReport] = []

  for fname, raw_mat in eval_outputs.items():
    finite_total = int(np.sum(np.isfinite(raw_mat)))
    if finite_total == 0:
      raise ValueError(
        f"Factor '{fname}' produced 0 finite values across evaluation window "
        f"[{start_date}, {end_date}] ({eval_panel.n_objs} stocks x {eval_panel.bars} bars)."
      )

    direction = factor.get_direction(fname)
    oriented = raw_mat * float(direction) if direction == -1 else raw_mat

    proc_mat = preprocess_factor(
      oriented,
      signal_mask=fwd_set.signal_mask,
      winsorize_method=winsorize,
      neutralize_modes=norm_neut,
      standardize_method=standardize,
      industry_codes=industry_codes,
      log_mktcap=log_mktcap,
    )

    if int(np.sum(np.isfinite(proc_mat))) == 0:
      raise ValueError(
        f"Factor '{fname}' has 0 valid values after tradability filtering and preprocessing "
        f"(winsorize={winsorize}, neutralize={norm_neut}, standardize={standardize})."
      )

    # Coverage & constant cross-section diagnostics
    valid_per_bar = np.sum(np.isfinite(proc_mat), axis=0)
    cov_per_bar = valid_per_bar / float(max(1, eval_panel.n_objs))
    cov_mean = float(np.mean(cov_per_bar))
    cov_min = float(np.min(cov_per_bar))

    multi_cols = np.flatnonzero(valid_per_bar >= 2)
    if len(multi_cols) > 0:
      with np.errstate(all="ignore"):
        cs_std = np.nanstd(proc_mat[:, multi_cols], axis=0)
      constant_cs_days = int(np.sum(cs_std <= 1e-12))
    else:
      constant_cs_days = 0
    if constant_cs_days == len(multi_cols) and len(multi_cols) > 0:
      raise ValueError(
        f"Factor '{fname}' is constant across every cross-section in [{start_date}, {end_date}]."
      )

    # Market cap exposure diagnostic
    mktcap_corr: float | None = None
    if log_mktcap is not None:
      cap_ric = compute_rank_ic_series(proc_mat, log_mktcap, min_obs=min_obs)
      v_cap = cap_ric[np.isfinite(cap_ric)]
      if len(v_cap) > 0:
        mktcap_corr = float(np.mean(v_cap))

    ic_summaries: dict[int, ICSummary] = {}
    ic_series_map: dict[int, np.ndarray] = {}
    rank_ic_series_map: dict[int, np.ndarray] = {}

    for p in norm_periods:
      ic_s, ric_s = compute_ic_series(proc_mat, fwd_set.forward_returns[p], min_obs=min_obs)
      ic_series_map[p] = ic_s
      rank_ic_series_map[p] = ric_s
      ic_summaries[p] = summarize_ic(ic_s, ric_s, period=p)

    bins = assign_quantiles(proc_mat, n_quantiles=quantiles)
    qa_map = compute_quantile_analyses(
      proc_mat,
      fwd_set,
      norm_periods,
      n_quantiles=quantiles,
      cost_bps=cost_bps,
      benchmark_daily_ret=bench_daily_ret,
      bins=bins,
    )

    turnover_stats = compute_turnover_and_autocorr(
      proc_mat,
      n_quantiles=quantiles,
      min_obs=min_obs,
      bins=bins,
    )

    annual_rows = compute_annual_breakdown(
      dates,
      ic_series_map[prim_p],
      rank_ic_series_map[prim_p],
      qa_map[prim_p].long_excess_nav,
      period=prim_p,
    )

    holdout_res: dict[str, Any] | None = None
    if holdout:
      holdout_res = compute_holdout_breakdown(
        dates,
        proc_mat,
        fwd_set,
        prim_p,
        holdout,
        n_quantiles=quantiles,
        cost_bps=cost_bps,
        min_obs=min_obs,
      )

    ind_rows: list[dict[str, Any]] = []
    if by_industry and industry_map is not None:
      ind_rows = compute_industry_breakdown(
        proc_mat,
        fwd_set.forward_returns[prim_p],
        industry_names,
        period=prim_p,
      )

    alerts = _build_warnings(
      name=fname,
      direction=direction,
      primary_ic=ic_summaries[prim_p],
      primary_qa=qa_map[prim_p],
      coverage_mean=cov_mean,
      mktcap_corr=mktcap_corr,
      neutralize=norm_neut,
      top_daily_turnover=turnover_stats["top_daily_turnover"],
      holdout_breakdown=holdout_res,
      constant_cs_days=constant_cs_days,
      bars=eval_panel.bars,
    )

    expr_str = ""
    if isinstance(factor, ExpressionFactor) and fname in factor.compiled:
      expr_str = factor.compiled[fname].expression

    diagnostics: dict[str, Any] = {
      "coverage_mean": cov_mean,
      "coverage_min": cov_min,
      "constant_cs_days": constant_cs_days,
      "suspended_ratio": fwd_set.suspended_ratio,
      "limit_filtered_ratio": fwd_set.limit_filtered_ratio,
      "mktcap_corr": mktcap_corr,
      "causality_check": causality_res,
      "warnings": alerts,
    }
    if custom_diag:
      diagnostics["custom"] = custom_diag

    processed_map[fname] = proc_mat
    reports.append(
      FactorReport(
        name=fname,
        dates=dates,
        n_objs=eval_panel.n_objs,
        bars=eval_panel.bars,
        start_date=start_date,
        end_date=end_date,
        periods=norm_periods,
        primary_period=prim_p,
        quantiles=quantiles,
        entry=entry,
        winsorize=winsorize,
        standardize=standardize,
        neutralize=norm_neut,
        cost_bps=cost_bps,
        direction=direction,
        benchmark=benchmark_name,
        ic_summaries=ic_summaries,
        ic_series=ic_series_map,
        rank_ic_series=rank_ic_series_map,
        quantile_analyses=qa_map,
        turnover=turnover_stats,
        annual_breakdown=annual_rows,
        holdout_breakdown=holdout_res,
        industry_breakdown=ind_rows,
        diagnostics=diagnostics,
        expression=expr_str,
        author=author,
      )
    )

  if len(reports) == 1:
    return reports[0]

  corr_mat = compute_factor_correlation_matrix(processed_map, min_obs=min_obs)
  return MultiFactorReport(
    reports=reports,
    correlation_matrix=corr_mat,
    primary_period=prim_p,
  )


def evaluate_factor(
  factor: Factor | str,
  objs: Sequence[str],
  start: str,
  end: str,
  *,
  msd_host: str | None = None,
  benchmark: str = "SH000300",
  factor_kwargs: dict[str, Any] | None = None,
  fin_fields: Sequence[str] | None = None,
  fin_only_year: bool = False,
  fin_lookback_days: int | None = None,
  warmup_bars: int | None = None,
  periods: Sequence[int] = (1, 5, 10, 20),
  primary_period: int | None = None,
  quantiles: int = 5,
  entry: str = "next_open",
  exclude_limit: bool = True,
  min_listed_days: int = 0,
  winsorize: str = "none",
  standardize: str = "none",
  neutralize: Sequence[str] = (),
  cost_bps: float = 0.0,
  holdout: str | None = None,
  causality_check: bool = True,
  min_obs: int = 10,
  by_industry: bool = False,
  industry_category: str = "申万行业",
) -> FactorReport | MultiFactorReport:
  """Load market/financial data from MSD and evaluate a factor over ``[start, end]``.

  Uses ``benchmark`` as the trading-calendar baseline during MSD loading, then
  separates the benchmark series out of the cross-sectional stock universe so
  index bars never contaminate IC or quantile rankings.
  """
  if isinstance(factor, str):
    ext = os.path.splitext(factor)[1].lower()
    if ext == ".py":
      factor_inst = load_factor(factor, factor_kwargs, strict_args=True)
    elif ext in (".txt", ".json"):
      exprs, dirs = load_expression_file(factor)
      factor_inst = ExpressionFactor(exprs, direction=dirs)
    else:
      factor_inst = ExpressionFactor(factor)
  else:
    factor_inst = factor

  eval_objs = list(dict.fromkeys(objs))
  if not eval_objs:
    raise ValueError(
      "No candidate stocks provided for factor evaluation. "
      "Pass a stock universe via `objs` / `-s` (e.g. `-s A_STOCKS_EXCLUDE_ST`)."
    )

  # Merge financial fields & warmup requirements
  merged_fin: list[str] = []
  for src in (fin_fields, factor_inst.fin_fields):
    if src:
      for f in src:
        f_clean = str(f).strip()
        if f_clean and f_clean not in merged_fin:
          merged_fin.append(f_clean)

  eff_fin_only_year = bool(fin_only_year or factor_inst.fin_only_year)
  eff_fin_lookback = (
    fin_lookback_days if fin_lookback_days is not None else factor_inst.fin_lookback_days
  )
  eff_warmup = int(warmup_bars) if warmup_bars is not None else int(factor_inst.warmup_bars)

  # Prepend benchmark as calendar alignment baseline (objs[0] in load_data)
  bench_sym = benchmark.strip() if benchmark else ""
  request_objs = [s for s in eval_objs if s != bench_sym]
  if not request_objs:
    raise ValueError(
      f"Stock universe contains only benchmark '{bench_sym}'; no evaluable stocks remain."
    )
  if bench_sym:
    load_objs = [bench_sym] + request_objs
  else:
    load_objs = list(request_objs)

  host = msd_host or os.environ.get("MSD_HOST", "http://localhost:50510")
  from mbt.run.msd import load_data

  dp, loaded_objs = load_data(
    host,
    load_objs,
    start,
    end,
    fin_fields=merged_fin if merged_fin else None,
    fin_only_year=eff_fin_only_year,
    warmup_bars=eff_warmup,
    fin_lookback_days=eff_fin_lookback,
  )

  full_panel = dp.to_panel()
  bench_close: np.ndarray | None = None
  bench_open: np.ndarray | None = None

  if bench_sym and bench_sym in full_panel.objs:
    b_idx = full_panel.pick(bench_sym)
    bench_close = np.array(full_panel.column("close")[b_idx, :], dtype=np.float64, copy=True)
    bench_open = np.array(full_panel.column("open")[b_idx, :], dtype=np.float64, copy=True)
    stock_objs = [s for s in loaded_objs if s != bench_sym]
    if not stock_objs:
      raise ValueError(
        f"After excluding benchmark '{bench_sym}', 0 loaded stocks remain in MSD result."
      )
    stock_panel = full_panel.slice(0, full_panel.bars, objs=stock_objs)
  else:
    stock_panel = full_panel

  norm_neut = tuple(
    dict.fromkeys(m.strip().lower() for m in neutralize if m and m.strip().lower() != "none")
  )
  need_ind = ("industry" in norm_neut) or by_industry or bool(factor_inst.needs_industry)
  industry_map: dict[str, str] | None = None
  if need_ind:
    from mbt.shared.stocks import get_stock_industry_map

    industry_map = get_stock_industry_map(stock_panel.objs, category=industry_category)

  return evaluate_panel(
    factor_inst,
    stock_panel,
    periods=periods,
    primary_period=primary_period,
    quantiles=quantiles,
    entry=entry,
    exclude_limit=exclude_limit,
    min_listed_days=min_listed_days,
    winsorize=winsorize,
    standardize=standardize,
    neutralize=norm_neut,
    cost_bps=cost_bps,
    warmup_bars=int(getattr(dp, "warmup_bars", 0)),
    holdout=holdout,
    causality_check=causality_check,
    min_obs=min_obs,
    industry_map=industry_map,
    by_industry=by_industry,
    benchmark_name=bench_sym,
    benchmark_close=bench_close,
    benchmark_open=bench_open,
  )
